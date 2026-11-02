package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/gorilla/websocket"
	"github.com/redis/go-redis/v9"
)

//go:embed static/*
var staticFiles embed.FS

// ── Qdrant types ─────────────────────────────────────────────────────────────

type qdrantClient struct {
	base string
	key  string
	http *http.Client
}

type scrollRequest struct {
	Limit       int         `json:"limit"`
	WithPayload bool        `json:"with_payload"`
	WithVectors bool        `json:"with_vectors"`
	Offset      interface{} `json:"offset,omitempty"`
}

type scrollResult struct {
	Result *scrollResultInner `json:"result"`
	Status string             `json:"status"`
}

type scrollResultInner struct {
	Points   []QPoint    `json:"points"`
	NextPage interface{} `json:"next_page_offset"`
}

type QPoint struct {
	ID      interface{}            `json:"id"`
	Vector  interface{}            `json:"vector"`
	Payload map[string]interface{} `json:"payload"`
}

type collectionInfo struct {
	Result struct {
		PointsCount uint64 `json:"points_count"`
		Config      struct {
			Params struct {
				Vectors struct {
					Size     uint64 `json:"size"`
					Distance string `json:"distance"`
				} `json:"vectors"`
			} `json:"params"`
		} `json:"config"`
	} `json:"result"`
}

type collectionsListResult struct {
	Result struct {
		Collections []struct {
			Name string `json:"name"`
		} `json:"collections"`
	} `json:"result"`
}

type pointsByIDRequest struct {
	IDs         []interface{} `json:"ids"`
	WithPayload bool          `json:"with_payload"`
	WithVector  bool          `json:"with_vector"`
}

type pointsByIDResult struct {
	Result []QPoint `json:"result"`
	Status string   `json:"status"`
}

type searchPointsRequest struct {
	Vector         interface{} `json:"vector"`
	Limit          int         `json:"limit"`
	WithPayload    bool        `json:"with_payload"`
	WithVectors    bool        `json:"with_vectors,omitempty"`
	ScoreThreshold *float64    `json:"score_threshold,omitempty"`
}

type searchPointHit struct {
	ID      interface{}            `json:"id"`
	Score   float64                `json:"score"`
	Payload map[string]interface{} `json:"payload"`
}

type searchPointsResult struct {
	Result []searchPointHit `json:"result"`
	Status string           `json:"status"`
}

type namedSearchVector struct {
	Name   string    `json:"name"`
	Vector []float64 `json:"vector"`
}

type similarNeighbor struct {
	ID      interface{}            `json:"id"`
	Score   float64                `json:"score"`
	Payload map[string]interface{} `json:"payload"`
}

type similarResponse struct {
	SelectedID       interface{}       `json:"selected_id"`
	Collection       string            `json:"collection"`
	Limit            int               `json:"limit"`
	SimilarityRadius *float64          `json:"similarity_radius,omitempty"`
	VectorName       string            `json:"vector_name,omitempty"`
	Neighbors        []similarNeighbor `json:"neighbors"`
}

type highlightRequest struct {
	Collection string        `json:"collection"`
	IDs        []interface{} `json:"ids"`
	FocusID    interface{}   `json:"focus_id,omitempty"`
}

type highlightEvent struct {
	EventID    string        `json:"event_id"`
	Collection string        `json:"collection"`
	IDs        []interface{} `json:"ids"`
	FocusID    interface{}   `json:"focus_id,omitempty"`
	CreatedAt  int64         `json:"created_at"`
}

type streamEvent struct {
	Type       string      `json:"type"`
	Collection string      `json:"collection,omitempty"`
	Timestamp  int64       `json:"timestamp"`
	Payload    interface{} `json:"payload,omitempty"`
}

type streamSubscriber struct {
	Collection string
	Channel    chan []byte
}

type redisStreamRecorder struct {
	client *redis.Client
	key    string
	maxLen int64
}

func newRedisStreamRecorder(redisURL, key string, maxLen int64) (*redisStreamRecorder, error) {
	redisURL = strings.TrimSpace(redisURL)
	key = strings.TrimSpace(key)
	if redisURL == "" || key == "" {
		return nil, nil
	}
	opts, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("invalid redis stream recorder url: %w", err)
	}
	client := redis.NewClient(opts)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		_ = client.Close()
		return nil, fmt.Errorf("redis stream recorder ping failed: %w", err)
	}
	if maxLen <= 0 {
		maxLen = 20000
	}
	return &redisStreamRecorder{client: client, key: key, maxLen: maxLen}, nil
}

func (r *redisStreamRecorder) append(ctx context.Context, evt streamEvent, encoded []byte) error {
	if r == nil || r.client == nil || strings.TrimSpace(r.key) == "" || len(encoded) == 0 {
		return nil
	}
	values := map[string]interface{}{
		"event":      string(encoded),
		"type":       evt.Type,
		"collection": evt.Collection,
		"timestamp":  evt.Timestamp,
	}
	args := &redis.XAddArgs{Stream: r.key, MaxLen: r.maxLen, Approx: true, Values: values}
	return r.client.XAdd(ctx, args).Err()
}

func (r *redisStreamRecorder) replay(ctx context.Context, count int, collection string) ([][]byte, error) {
	if r == nil || r.client == nil || strings.TrimSpace(r.key) == "" || count <= 0 {
		return nil, nil
	}
	entries, err := r.client.XRevRangeN(ctx, r.key, "+", "-", int64(count)).Result()
	if err != nil {
		return nil, err
	}
	collection = strings.TrimSpace(collection)
	out := make([][]byte, 0, len(entries))
	for i := len(entries) - 1; i >= 0; i-- {
		entry := entries[i]
		if collection != "" {
			entryCollection := strings.TrimSpace(fmt.Sprintf("%v", entry.Values["collection"]))
			if entryCollection != "" && entryCollection != collection {
				continue
			}
		}
		rawEvent, ok := entry.Values["event"]
		if !ok {
			continue
		}
		text := strings.TrimSpace(fmt.Sprintf("%v", rawEvent))
		if text == "" {
			continue
		}
		out = append(out, []byte(text))
	}
	return out, nil
}

type streamHub struct {
	mu       sync.RWMutex
	nextID   int
	subs     map[int]streamSubscriber
	recorder *redisStreamRecorder
}

func newStreamHub(recorder *redisStreamRecorder) *streamHub {
	return &streamHub{subs: make(map[int]streamSubscriber), recorder: recorder}
}

func (h *streamHub) subscribe(collection string) (int, chan []byte) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.nextID++
	id := h.nextID
	ch := make(chan []byte, 64)
	h.subs[id] = streamSubscriber{Collection: strings.TrimSpace(collection), Channel: ch}
	return id, ch
}

func (h *streamHub) unsubscribe(id int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	sub, ok := h.subs[id]
	if !ok {
		return
	}
	delete(h.subs, id)
	close(sub.Channel)
}

func (h *streamHub) replay(ctx context.Context, count int, collection string) ([][]byte, error) {
	if h == nil {
		return nil, nil
	}
	return h.recorder.replay(ctx, count, collection)
}

func streamEventType(payload []byte) string {
	if len(payload) == 0 {
		return "event"
	}
	var envelope streamEvent
	if err := json.Unmarshal(payload, &envelope); err != nil {
		return "event"
	}
	eventType := strings.TrimSpace(envelope.Type)
	if eventType == "" {
		return "event"
	}
	return eventType
}

func (h *streamHub) publish(eventType, collection string, payload interface{}) {
	if h == nil {
		return
	}
	evt := streamEvent{
		Type:       strings.TrimSpace(eventType),
		Collection: strings.TrimSpace(collection),
		Timestamp:  time.Now().UnixMilli(),
		Payload:    payload,
	}
	if evt.Type == "" {
		evt.Type = "event"
	}
	encoded, err := json.Marshal(evt)
	if err != nil {
		return
	}
	if err := h.recorder.append(context.Background(), evt, encoded); err != nil {
		log.Printf("stream recorder append error: %v", err)
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	for _, sub := range h.subs {
		if sub.Collection != "" && sub.Collection != evt.Collection {
			continue
		}
		select {
		case sub.Channel <- encoded:
		default:
		}
	}
}

// ── API response types ────────────────────────────────────────────────────────

type PointsResponse struct {
	Points       []PointData     `json:"points"`
	Total        int             `json:"total"`
	Projection   *ProjectionMeta `json:"projection,omitempty"`
	Incremental  bool            `json:"incremental,omitempty"`
	AppendFrom   int             `json:"append_from,omitempty"`
	CachedResult bool            `json:"cached_result,omitempty"`
}

type ProjectionMeta struct {
	Method string           `json:"method"`
	Axes   []ProjectionAxis `json:"axes"`
}

type ProjectionAxis struct {
	Component         string  `json:"component"`
	VarianceExplained float64 `json:"variance_explained"`
}

type PointData struct {
	ID      interface{}            `json:"id"`
	X       float64                `json:"x"`
	Y       float64                `json:"y"`
	Z       float64                `json:"z"`
	Payload map[string]interface{} `json:"payload"`
}

type CollectionMeta struct {
	Name            string `json:"name"`
	PointsCount     uint64 `json:"points_count"`
	VectorSize      uint64 `json:"vector_size"`
	DistanceMetric  string `json:"distance_metric,omitempty"`
	ProjectionReady bool   `json:"projection_ready"`
	ProjectionNote  string `json:"projection_note,omitempty"`
}

type loadProgress struct {
	Loaded  int    `json:"loaded"`
	Limit   int    `json:"limit"`
	Percent int    `json:"percent"`
	Status  string `json:"status"`
	Done    bool   `json:"done"`
	Error   string `json:"error,omitempty"`
}

type progressReporter func(progressID string, progress loadProgress)

var highlightEvents sync.Map
var wsUpgrader = websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}

// defaultCollection is the fallback when a request doesn't specify ?collection=.
// Change this in one spot rather than chasing fallbacks across handlers.
var defaultCollection = "meta_reflections"

type projectionCache interface {
	Get(ctx context.Context, key string) (*PointsResponse, bool, error)
	Set(ctx context.Context, key string, value *PointsResponse) error
}

type redisProjectionCache struct {
	client *redis.Client
	ttl    time.Duration
}

func newRedisProjectionCache(redisURL string, ttl time.Duration) (*redisProjectionCache, error) {
	opts, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("invalid redis url: %w", err)
	}
	client := redis.NewClient(opts)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("redis ping failed: %w", err)
	}
	return &redisProjectionCache{client: client, ttl: ttl}, nil
}

func (c *redisProjectionCache) Get(ctx context.Context, key string) (*PointsResponse, bool, error) {
	raw, err := c.client.Get(ctx, key).Result()
	if err == redis.Nil {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	var out PointsResponse
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return nil, false, err
	}
	out.CachedResult = true
	return &out, true, nil
}

func (c *redisProjectionCache) Set(ctx context.Context, key string, value *PointsResponse) error {
	if value == nil {
		return nil
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return c.client.Set(ctx, key, encoded, c.ttl).Err()
}

func projectionCacheKey(collection string, limit int, projectionMethod, vectorName string) string {
	vectorName = strings.TrimSpace(vectorName)
	if vectorName == "" {
		vectorName = "_default"
	}
	return fmt.Sprintf("vectorview:projection:v1:%s:%d:%s:%s", collection, limit, projectionMethod, vectorName)
}

// ── Disk projection cache ─────────────────────────────────────────────────────

type diskProjectionCache struct {
	dir      string
	ttl      time.Duration
	maxBytes int64
	writes   atomic.Int64
}

type diskCacheEntry struct {
	StoredAt int64           `json:"stored_at"`
	Data     *PointsResponse `json:"data"`
}

func newDiskProjectionCache(dir string, ttl time.Duration, maxBytes int64) (*diskProjectionCache, error) {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return nil, nil
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("resolve cache dir: %w", err)
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return nil, fmt.Errorf("create cache dir: %w", err)
	}
	if ttl <= 0 {
		ttl = 10 * time.Minute
	}
	if maxBytes <= 0 {
		maxBytes = 512 << 20
	}
	cache := &diskProjectionCache{dir: abs, ttl: ttl, maxBytes: maxBytes}
	cache.prune()
	return cache, nil
}

func (c *diskProjectionCache) path(key string) string {
	sum := sha256.Sum256([]byte(key))
	return filepath.Join(c.dir, hex.EncodeToString(sum[:])+".json")
}

func (c *diskProjectionCache) Get(ctx context.Context, key string) (*PointsResponse, bool, error) {
	if c == nil {
		return nil, false, nil
	}
	raw, err := os.ReadFile(c.path(key))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, false, nil
		}
		return nil, false, err
	}
	var entry diskCacheEntry
	if err := json.Unmarshal(raw, &entry); err != nil {
		_ = os.Remove(c.path(key))
		return nil, false, nil
	}
	if entry.Data == nil {
		return nil, false, nil
	}
	if time.Since(time.UnixMilli(entry.StoredAt)) > c.ttl {
		_ = os.Remove(c.path(key))
		return nil, false, nil
	}
	out := *entry.Data
	out.CachedResult = true
	return &out, true, nil
}

func (c *diskProjectionCache) Set(ctx context.Context, key string, value *PointsResponse) error {
	if c == nil || value == nil {
		return nil
	}
	encoded, err := json.Marshal(diskCacheEntry{StoredAt: time.Now().UnixMilli(), Data: value})
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(c.dir, "projection-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(encoded); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	if err := os.Rename(tmpName, c.path(key)); err != nil {
		os.Remove(tmpName)
		return err
	}
	if c.writes.Add(1)%64 == 0 {
		c.prune()
	}
	return nil
}

func (c *diskProjectionCache) prune() {
	if c == nil {
		return
	}
	entries, err := os.ReadDir(c.dir)
	if err != nil {
		return
	}
	type cachedFile struct {
		path    string
		modTime time.Time
		size    int64
	}
	files := make([]cachedFile, 0, len(entries))
	var total int64
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		full := filepath.Join(c.dir, entry.Name())
		if time.Since(info.ModTime()) > c.ttl {
			_ = os.Remove(full)
			continue
		}
		files = append(files, cachedFile{path: full, modTime: info.ModTime(), size: info.Size()})
		total += info.Size()
	}
	if total <= c.maxBytes {
		return
	}
	sort.Slice(files, func(i, j int) bool { return files[i].modTime.Before(files[j].modTime) })
	for _, file := range files {
		if total <= c.maxBytes {
			return
		}
		if err := os.Remove(file.path); err == nil {
			total -= file.size
		}
	}
}

func (c *diskProjectionCache) label() string {
	if c == nil {
		return "disk"
	}
	return "disk:" + c.dir
}

type tieredProjectionCache struct {
	backends []projectionCache
}

func (c *tieredProjectionCache) Get(ctx context.Context, key string) (*PointsResponse, bool, error) {
	var firstErr error
	for _, backend := range c.backends {
		if backend == nil {
			continue
		}
		value, ok, err := backend.Get(ctx, key)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		if ok && value != nil {
			return value, true, nil
		}
	}
	return nil, false, firstErr
}

func (c *tieredProjectionCache) Set(ctx context.Context, key string, value *PointsResponse) error {
	var firstErr error
	for _, backend := range c.backends {
		if backend == nil {
			continue
		}
		if err := backend.Set(ctx, key, value); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// ── Projection concurrency control ────────────────────────────────────────────

var errProjectionBusy = errors.New("projection capacity exhausted")

type projectionGuard struct {
	slots        chan struct{}
	queueTimeout time.Duration
}

func defaultProjectionSlots() int {
	if raw := strings.TrimSpace(os.Getenv("VECTORVIEW_MAX_PROJECTIONS")); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			return n
		}
	}
	n := runtime.NumCPU() / 2
	if n < 1 {
		n = 1
	}
	return n
}

func defaultProjectionQueueTimeout() time.Duration {
	seconds, err := strconv.Atoi(strings.TrimSpace(envOr("VECTORVIEW_PROJECTION_QUEUE_SECONDS", "30")))
	if err != nil || seconds <= 0 {
		seconds = 30
	}
	return time.Duration(seconds) * time.Second
}

func newProjectionGuard(slots int, queueTimeout time.Duration) *projectionGuard {
	if slots <= 0 {
		slots = 1
	}
	if queueTimeout <= 0 {
		queueTimeout = 30 * time.Second
	}
	return &projectionGuard{slots: make(chan struct{}, slots), queueTimeout: queueTimeout}
}

func (g *projectionGuard) capacity() int {
	if g == nil {
		return 0
	}
	return cap(g.slots)
}

func (g *projectionGuard) inUse() int {
	if g == nil {
		return 0
	}
	return len(g.slots)
}

func (g *projectionGuard) acquire(ctx context.Context) error {
	if g == nil {
		return nil
	}
	select {
	case g.slots <- struct{}{}:
		return nil
	default:
	}
	timer := time.NewTimer(g.queueTimeout)
	defer timer.Stop()
	select {
	case g.slots <- struct{}{}:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("%w: %v", errProjectionBusy, ctx.Err())
	case <-timer.C:
		return errProjectionBusy
	}
}

func (g *projectionGuard) release() {
	if g == nil {
		return
	}
	<-g.slots
}

var projections = newProjectionGuard(defaultProjectionSlots(), defaultProjectionQueueTimeout())

type projectionFlight struct {
	done chan struct{}
	resp PointsResponse
	err  error
}

type projectionFlights struct {
	mu    sync.Mutex
	calls map[string]*projectionFlight
}

func newProjectionFlights() *projectionFlights {
	return &projectionFlights{calls: make(map[string]*projectionFlight)}
}

func (g *projectionFlights) do(key string, fn func() (PointsResponse, error)) (PointsResponse, error) {
	if g == nil || strings.TrimSpace(key) == "" {
		return fn()
	}
	g.mu.Lock()
	if existing, ok := g.calls[key]; ok {
		g.mu.Unlock()
		<-existing.done
		return existing.resp, existing.err
	}
	flight := &projectionFlight{done: make(chan struct{})}
	g.calls[key] = flight
	g.mu.Unlock()

	func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				flight.err = fmt.Errorf("projection panic: %v", recovered)
			}
			g.mu.Lock()
			delete(g.calls, key)
			g.mu.Unlock()
			close(flight.done)
		}()
		flight.resp, flight.err = fn()
	}()
	return flight.resp, flight.err
}

var projectionInFlight = newProjectionFlights()

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ── Qdrant client ─────────────────────────────────────────────────────────────

func newQdrant(base, key string) *qdrantClient {
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	if base == "" {
		base = "http://localhost:6333"
	}
	return &qdrantClient{base: base, key: key, http: &http.Client{Timeout: 120 * time.Second}}
}

func (q *qdrantClient) do(ctx context.Context, method, path string, body any) ([]byte, int, error) {
	var r io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, 0, fmt.Errorf("marshal: %w", err)
		}
		r = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, q.base+path, r)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if q.key != "" {
		req.Header.Set("api-key", q.key)
	}
	resp, err := q.http.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	return raw, resp.StatusCode, nil
}

func (q *qdrantClient) listCollections(ctx context.Context) ([]string, error) {
	raw, _, err := q.do(ctx, http.MethodGet, "/collections", nil)
	if err != nil {
		return nil, err
	}
	var res collectionsListResult
	if err := json.Unmarshal(raw, &res); err != nil {
		return nil, fmt.Errorf("listCollections decode: %w — body: %s", err, string(raw))
	}
	names := make([]string, 0)
	for _, c := range res.Result.Collections {
		names = append(names, c.Name)
	}
	return names, nil
}

func (q *qdrantClient) collectionMeta(ctx context.Context, name string) (CollectionMeta, error) {
	raw, _, err := q.do(ctx, http.MethodGet, "/collections/"+name, nil)
	if err != nil {
		return CollectionMeta{}, err
	}
	var info collectionInfo
	if err := json.Unmarshal(raw, &info); err != nil {
		return CollectionMeta{}, fmt.Errorf("collectionMeta decode: %w", err)
	}
	return CollectionMeta{
		Name:           name,
		PointsCount:    info.Result.PointsCount,
		VectorSize:     info.Result.Config.Params.Vectors.Size,
		DistanceMetric: info.Result.Config.Params.Vectors.Distance,
	}, nil
}

func (q *qdrantClient) collectionProjectionStatus(ctx context.Context, name string) (bool, string) {
	req := scrollRequest{Limit: 64, WithPayload: false, WithVectors: true}
	raw, status, err := q.do(ctx, http.MethodPost, "/collections/"+name+"/points/scroll", req)
	if err != nil {
		return false, err.Error()
	}
	if status >= 400 {
		return false, fmt.Sprintf("qdrant %d", status)
	}

	var res scrollResult
	if err := json.Unmarshal(raw, &res); err != nil {
		return false, fmt.Sprintf("decode: %v", err)
	}
	if res.Result == nil || len(res.Result.Points) == 0 {
		return true, "empty collection"
	}
	if chooseProjectionDimension(res.Result.Points) == 0 {
		return false, "no dense vectors"
	}
	return true, "ok"
}

func (q *qdrantClient) detectCollectionEmbedding(ctx context.Context, collection string) (provider, model string, dims int) {
	meta, err := q.collectionMeta(ctx, collection)
	if err == nil {
		dims = int(meta.VectorSize)
	}
	req := scrollRequest{Limit: 1, WithPayload: true, WithVectors: false}
	raw, status, err := q.do(ctx, http.MethodPost, "/collections/"+collection+"/points/scroll", req)
	if err != nil || status >= 400 {
		return "", "", dims
	}
	var res scrollResult
	if err := json.Unmarshal(raw, &res); err != nil || res.Result == nil || len(res.Result.Points) == 0 {
		return "", "", dims
	}
	p := res.Result.Points[0]
	if p.Payload != nil {
		if prov, ok := p.Payload["embedding_provider"].(string); ok && strings.TrimSpace(prov) != "" {
			provider = strings.ToLower(strings.TrimSpace(prov))
		}
		if mod, ok := p.Payload["embedding_model"].(string); ok && strings.TrimSpace(mod) != "" {
			model = strings.TrimSpace(mod)
		}
		if d, ok := p.Payload["embedding_dimensions"].(float64); ok && d > 0 {
			dims = int(d)
		}
	}
	return provider, model, dims
}

func (q *qdrantClient) getPointByID(ctx context.Context, collection, pointID string) (*QPoint, error) {
	idCandidates := []interface{}{pointID}
	if n, err := strconv.ParseInt(pointID, 10, 64); err == nil {
		idCandidates = append(idCandidates, n)
	}
	if n, err := strconv.ParseUint(pointID, 10, 64); err == nil {
		idCandidates = append(idCandidates, n)
	}

	var lastErr error
	hadSuccessfulLookup := false
	for _, idCandidate := range idCandidates {
		req := pointsByIDRequest{
			IDs:         []interface{}{idCandidate},
			WithPayload: true,
			WithVector:  true,
		}
		raw, status, err := q.do(ctx, http.MethodPost, "/collections/"+collection+"/points", req)
		if err != nil {
			lastErr = err
			continue
		}
		if status >= 400 {
			lastErr = fmt.Errorf("qdrant %d: %s", status, string(raw))
			continue
		}
		hadSuccessfulLookup = true
		var res pointsByIDResult
		if err := json.Unmarshal(raw, &res); err != nil {
			lastErr = fmt.Errorf("decode point lookup: %w", err)
			continue
		}
		if len(res.Result) > 0 {
			return &res.Result[0], nil
		}
	}

	if hadSuccessfulLookup {
		return nil, nil
	}
	if lastErr != nil {
		return nil, lastErr
	}
	return nil, nil
}

func (q *qdrantClient) fetchPointsWindow(ctx context.Context, collection string, start, count int) ([]QPoint, error) {
	if count <= 0 {
		return []QPoint{}, nil
	}
	offset := interface{}(nil)
	skipped := 0
	out := make([]QPoint, 0, count)

	for len(out) < count {
		req := scrollRequest{
			Limit:       minInt(200, max(32, count-len(out))),
			WithPayload: true,
			WithVectors: true,
			Offset:      offset,
		}
		raw, status, err := q.do(ctx, http.MethodPost, "/collections/"+collection+"/points/scroll", req)
		if err != nil {
			return nil, err
		}
		if status >= 400 {
			return nil, fmt.Errorf("qdrant %d: %s", status, string(raw))
		}
		var res scrollResult
		if err := json.Unmarshal(raw, &res); err != nil {
			return nil, fmt.Errorf("decode scroll window: %w", err)
		}
		if res.Result == nil || len(res.Result.Points) == 0 {
			break
		}

		points := res.Result.Points
		if skipped+len(points) <= start {
			skipped += len(points)
			offset = res.Result.NextPage
			if offset == nil {
				break
			}
			continue
		}

		begin := 0
		if start > skipped {
			begin = start - skipped
		}
		for i := begin; i < len(points) && len(out) < count; i++ {
			out = append(out, points[i])
		}
		skipped += len(points)
		offset = res.Result.NextPage
		if offset == nil {
			break
		}
	}

	return out, nil
}

func parseAppendFrom(raw string) (int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, nil
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value < 0 {
		return 0, fmt.Errorf("invalid append_from: %q", raw)
	}
	return value, nil
}

func parseSimilarityLimit(r *http.Request, fallback int) (int, error) {
	limit := fallback
	if raw := strings.TrimSpace(r.URL.Query().Get("limit")); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n <= 0 {
			return 0, fmt.Errorf("invalid limit: %q", raw)
		}
		limit = n
	}
	if r.Method == http.MethodPost && r.Body != nil {
		var body struct {
			Limit int `json:"limit"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil && err != io.EOF {
			return 0, fmt.Errorf("invalid request body: %w", err)
		}
		if body.Limit > 0 {
			limit = body.Limit
		}
	}
	if limit > 500 {
		limit = 500
	}
	return limit, nil
}

func parseSimilarityRadius(r *http.Request, fallback float64) (float64, error) {
	radius := fallback
	if raw := strings.TrimSpace(r.URL.Query().Get("radius")); raw != "" {
		v, err := strconv.ParseFloat(raw, 64)
		if err != nil || v < 0 || v > 1 {
			return 0, fmt.Errorf("invalid radius: %q", raw)
		}
		radius = v
	}
	if r.Method == http.MethodPost && r.Body != nil {
		var body struct {
			Radius *float64 `json:"radius"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil && err != io.EOF {
			return 0, fmt.Errorf("invalid request body: %w", err)
		}
		if body.Radius != nil {
			v := *body.Radius
			if v < 0 || v > 1 {
				return 0, fmt.Errorf("invalid radius: %v", v)
			}
			radius = v
		}
	}
	return radius, nil
}

func parseProjectionMethod(raw string) (string, error) {
	method := strings.ToLower(strings.TrimSpace(raw))
	if method == "" {
		method = "pca"
	}
	switch method {
	case "pca", "random", "tsne", "umap":
		return method, nil
	default:
		return "", fmt.Errorf("invalid projection: %q (expected pca, random, tsne, umap)", raw)
	}
}

func embedQueryWithOllama(ctx context.Context, ollamaURL, model, query string) ([]float64, error) {
	if strings.TrimSpace(model) == "" {
		return nil, fmt.Errorf("VECTORVIEW_EMBED_MODEL is required")
	}
	base := strings.TrimRight(strings.TrimSpace(ollamaURL), "/")
	if base == "" {
		return nil, fmt.Errorf("VECTORVIEW_OLLAMA_URL is required")
	}

	payload, err := json.Marshal(map[string]interface{}{
		"model": model,
		"input": query,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal ollama payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/api/embed", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("ollama %d: %s", resp.StatusCode, string(body))
	}

	var parsed struct {
		Embeddings [][]float64 `json:"embeddings"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("decode ollama embed response: %w", err)
	}
	if len(parsed.Embeddings) == 0 || len(parsed.Embeddings[0]) == 0 {
		return nil, fmt.Errorf("ollama embed returned no vectors")
	}
	return parsed.Embeddings[0], nil
}

func embedQueryWithOpenRouter(ctx context.Context, openrouterURL, apiKey, model, query string, dims int) ([]float64, error) {
	apiKey = strings.TrimSpace(apiKey)
	if apiKey == "" {
		return nil, fmt.Errorf("OPENROUTER_API_KEY is required for openrouter embeddings")
	}
	model = strings.TrimSpace(model)
	if model == "" {
		model = "google/gemini-embedding-2-preview"
	}
	base := strings.TrimRight(strings.TrimSpace(openrouterURL), "/")
	if base == "" {
		base = "https://openrouter.ai/api/v1"
	}

	payloadMap := map[string]interface{}{
		"model": model,
		"input": query,
	}
	if dims > 0 {
		payloadMap["dimensions"] = dims
	}

	payload, err := json.Marshal(payloadMap)
	if err != nil {
		return nil, fmt.Errorf("marshal openrouter payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/embeddings", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Title", "VectorView")

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("openrouter %d: %s", resp.StatusCode, string(body))
	}

	var parsed struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("decode openrouter response: %w", err)
	}
	if len(parsed.Data) == 0 || len(parsed.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("openrouter returned no embeddings")
	}
	return parsed.Data[0].Embedding, nil
}

func embedQueryWithOpenAI(ctx context.Context, openaiURL, apiKey, model, query string, dims int) ([]float64, error) {
	apiKey = strings.TrimSpace(apiKey)
	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY is required for openai embeddings")
	}
	model = strings.TrimSpace(model)
	if model == "" {
		model = "text-embedding-3-small"
	}
	base := strings.TrimRight(strings.TrimSpace(openaiURL), "/")
	if base == "" {
		base = "https://api.openai.com/v1"
	}

	payloadMap := map[string]interface{}{
		"model": model,
		"input": query,
	}
	if dims > 0 {
		payloadMap["dimensions"] = dims
	}

	payload, err := json.Marshal(payloadMap)
	if err != nil {
		return nil, fmt.Errorf("marshal openai payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/embeddings", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("openai %d: %s", resp.StatusCode, string(body))
	}

	var parsed struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("decode openai response: %w", err)
	}
	if len(parsed.Data) == 0 || len(parsed.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("openai returned no embeddings")
	}
	return parsed.Data[0].Embedding, nil
}

func canonicalPointID(v interface{}) string {
	switch t := v.(type) {
	case string:
		return t
	case float64:
		if t == float64(int64(t)) {
			return strconv.FormatInt(int64(t), 10)
		}
		return strconv.FormatFloat(t, 'f', -1, 64)
	case int:
		return strconv.Itoa(t)
	case int64:
		return strconv.FormatInt(t, 10)
	case json.Number:
		return t.String()
	default:
		return fmt.Sprintf("%v", t)
	}
}

func pickSearchVector(raw interface{}, preferredName string) (interface{}, string) {
	preferredName = strings.TrimSpace(preferredName)
	if vec := toFloatSlice(raw); len(vec) > 0 {
		return vec, ""
	}
	obj, ok := raw.(map[string]interface{})
	if !ok {
		return nil, ""
	}

	pickNamed := func(name string) (interface{}, string) {
		if name == "" {
			return nil, ""
		}
		rawValue, exists := obj[name]
		if !exists {
			return nil, ""
		}
		if vec := toFloatSlice(rawValue); len(vec) > 0 {
			if name == "vector" {
				return vec, ""
			}
			return namedSearchVector{Name: name, Vector: vec}, name
		}
		nested, ok := rawValue.(map[string]interface{})
		if !ok {
			return nil, ""
		}
		if vec := toFloatSlice(nested["vector"]); len(vec) > 0 {
			if name == "vector" {
				return vec, ""
			}
			return namedSearchVector{Name: name, Vector: vec}, name
		}
		return nil, ""
	}

	if preferred, preferredKey := pickNamed(preferredName); preferred != nil {
		return preferred, preferredKey
	}

	keys := make([]string, 0, len(obj))
	for key := range obj {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	for _, key := range keys {
		if vec, vecName := pickNamed(key); vec != nil {
			return vec, vecName
		}
	}
	return nil, ""
}

// ── GPU PCA via Python subprocess ─────────────────────────────────────────────

// pythonBin returns the best available python binary
func pythonBin() string {
	for _, candidate := range []string{"venv/bin/python3", "venv/bin/python"} {
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}
	for _, candidate := range []string{"python3", "python"} {
		if path, err := exec.LookPath(candidate); err == nil {
			return path
		}
	}
	return "python3"
}

// pcaScript returns the path to pca_gpu.py (same dir as binary, or cwd)
func pcaScript() string {
	// Try next to the executable first
	exe, err := os.Executable()
	if err == nil {
		candidate := filepath.Join(filepath.Dir(exe), "pca_gpu.py")
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}
	// Fall back to cwd
	return "pca_gpu.py"
}

func clampPercent(loaded, limit int) int {
	if limit <= 0 {
		return 0
	}
	percent := int(float64(loaded) / float64(limit) * 100.0)
	if percent < 0 {
		return 0
	}
	if percent > 100 {
		return 100
	}
	return percent
}

func parseFetchProgressLine(line string) (int, int, bool) {
	idx := strings.Index(line, "Fetched ")
	if idx == -1 {
		return 0, 0, false
	}
	segment := line[idx:]
	var loaded, limit int
	if _, err := fmt.Sscanf(segment, "Fetched %d / %d", &loaded, &limit); err != nil {
		return 0, 0, false
	}
	return loaded, limit, true
}

func writeSSEEvent(w io.Writer, eventType string, payload []byte) error {
	if _, err := fmt.Fprintf(w, "event: %s\n", strings.TrimSpace(eventType)); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", payload); err != nil {
		return err
	}
	return nil
}

func watchCollectionStream(ctx context.Context, q *qdrantClient, hub *streamHub, systemLabel, collection string, pollInterval time.Duration) {
	collection = strings.TrimSpace(collection)
	if collection == "" {
		return
	}
	knownCount := 0
	if meta, err := q.collectionMeta(ctx, collection); err == nil {
		knownCount = int(meta.PointsCount)
	}
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	warnedNotFound := false
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			meta, err := q.collectionMeta(ctx, collection)
			if err != nil {
				if !warnedNotFound {
					log.Printf("%s watch note: collection=%s not ready: %v", systemLabel, collection, err)
					warnedNotFound = true
				}
				continue
			}
			warnedNotFound = false
			currentCount := int(meta.PointsCount)
			if currentCount <= knownCount {
				continue
			}
			pointsToFetch := currentCount - knownCount
			if pointsToFetch > 128 {
				pointsToFetch = 128
			}
			points, err := q.fetchPointsWindow(ctx, collection, knownCount, pointsToFetch)
			if err != nil {
				log.Printf("%s watch error collection=%s: %v", systemLabel, collection, err)
				continue
			}
			if len(points) == 0 {
				knownCount = currentCount
				continue
			}
			knownCount += len(points)
			for _, point := range points {
				hub.publish("point_ingested", collection, map[string]interface{}{
					"id":      point.ID,
					"payload": point.Payload,
				})
			}
		}
	}
}

func startMetaBridgeWatchers(ctx context.Context, q *qdrantClient, hub *streamHub, collections []string, pollInterval time.Duration) {
	if len(collections) == 0 {
		return
	}
	for _, rawCollection := range collections {
		collection := strings.TrimSpace(rawCollection)
		if collection == "" {
			continue
		}
		go watchCollectionStream(ctx, q, hub, "meta_bridge", collection, pollInterval)
		log.Printf("meta_bridge watcher enabled for collection=%s", collection)
	}
}

func startFrontPocketWatchers(ctx context.Context, q *qdrantClient, hub *streamHub, collections []string, pollInterval time.Duration) {
	if len(collections) == 0 {
		return
	}
	for _, rawCollection := range collections {
		collection := strings.TrimSpace(rawCollection)
		if collection == "" {
			continue
		}
		go watchCollectionStream(ctx, q, hub, "frontpocket", collection, pollInterval)
		log.Printf("frontpocket watcher enabled for collection=%s", collection)
	}
}

func startRedisTelemetryBridge(ctx context.Context, redisURL, channel string, hub *streamHub) {
	redisURL = strings.TrimSpace(redisURL)
	channel = strings.TrimSpace(channel)
	if redisURL == "" || channel == "" {
		return
	}
	opts, err := redis.ParseURL(redisURL)
	if err != nil {
		log.Printf("redis telemetry bridge disabled: invalid redis url: %v", err)
		return
	}
	client := redis.NewClient(opts)
	pubsub := client.Subscribe(ctx, channel)
	go func() {
		defer pubsub.Close()
		defer client.Close()
		for {
			msg, err := pubsub.ReceiveMessage(ctx)
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				log.Printf("redis telemetry bridge receive error: %v", err)
				continue
			}
			var decoded interface{}
			if err := json.Unmarshal([]byte(msg.Payload), &decoded); err != nil {
				decoded = map[string]interface{}{"message": msg.Payload}
			}
			collection := ""
			if asMap, ok := decoded.(map[string]interface{}); ok {
				if v, ok := asMap["collection"]; ok {
					collection = strings.TrimSpace(fmt.Sprintf("%v", v))
				}
			}
			hub.publish("telemetry", collection, decoded)
		}
	}()
	log.Printf("redis telemetry bridge subscribed channel=%s", channel)
}

// runGPUPCA calls pca_gpu.py and returns projected points response
func runGPUPCA(parent context.Context, python string, report progressReporter, collection string, limit int, qdrantURL, projectionMethod, vectorName, progressID string) (PointsResponse, error) {
	py := python
	script := pcaScript()

	log.Printf("GPU PCA: %s %s %s %d %s", py, script, collection, limit, qdrantURL)
	report(progressID, loadProgress{Loaded: 0, Limit: limit, Percent: 0, Status: "starting", Done: false})

	if err := projections.acquire(parent); err != nil {
		report(progressID, loadProgress{Loaded: 0, Limit: limit, Percent: 0, Status: "failed", Done: true, Error: err.Error()})
		return PointsResponse{}, err
	}
	defer projections.release()

	ctx, cancel := context.WithTimeout(parent, 10*time.Minute)
	defer cancel()

	args := []string{
		script,
		collection,
		strconv.Itoa(limit),
		qdrantURL,
		projectionMethod,
	}
	if strings.TrimSpace(vectorName) != "" {
		args = append(args, vectorName)
	}
	cmd := exec.CommandContext(ctx, py, args...)
	cmd.Env = os.Environ()
	if qdrantKey := strings.TrimSpace(os.Getenv("QDRANT_API_KEY")); qdrantKey != "" {
		cmd.Env = append(cmd.Env, "QDRANT_API_KEY="+qdrantKey)
	}

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return PointsResponse{}, fmt.Errorf("stdout pipe: %w", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return PointsResponse{}, fmt.Errorf("stderr pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return PointsResponse{}, fmt.Errorf("start pca_gpu.py: %w", err)
	}

	stderrDone := make(chan struct{})
	go func() {
		defer close(stderrDone)
		scanner := bufio.NewScanner(stderrPipe)
		for scanner.Scan() {
			line := scanner.Text()
			log.Print(line)
			if loaded, progressLimit, ok := parseFetchProgressLine(line); ok {
				report(progressID, loadProgress{
					Loaded:  loaded,
					Limit:   progressLimit,
					Percent: clampPercent(loaded, progressLimit),
					Status:  "scrolling",
					Done:    false,
				})
				continue
			}
			if strings.Contains(line, "Fetch done") {
				report(progressID, loadProgress{Loaded: limit, Limit: limit, Percent: 100, Status: "projecting", Done: false})
			}
		}
	}()

	out, readErr := io.ReadAll(stdoutPipe)
	waitErr := cmd.Wait()
	<-stderrDone

	if readErr != nil {
		report(progressID, loadProgress{Loaded: 0, Limit: limit, Percent: 0, Status: "failed", Done: true, Error: readErr.Error()})
		return PointsResponse{}, fmt.Errorf("read pca_gpu.py output: %w", readErr)
	}
	if waitErr != nil {
		report(progressID, loadProgress{Loaded: 0, Limit: limit, Percent: 0, Status: "failed", Done: true, Error: waitErr.Error()})
		return PointsResponse{}, fmt.Errorf("pca_gpu.py failed: %w", waitErr)
	}

	var resp PointsResponse
	if err := json.Unmarshal(out, &resp); err != nil {
		report(progressID, loadProgress{Loaded: 0, Limit: limit, Percent: 0, Status: "failed", Done: true, Error: err.Error()})
		return PointsResponse{}, fmt.Errorf("pca_gpu.py bad JSON: %w — output: %.300s", err, string(out))
	}
	report(progressID, loadProgress{Loaded: len(resp.Points), Limit: limit, Percent: 100, Status: "done", Done: true})
	return resp, nil
}

func runProjectionOnPoints(parent context.Context, python string, points []QPoint, projectionMethod, vectorName string) (PointsResponse, error) {
	py := python
	script := pcaScript()

	if err := projections.acquire(parent); err != nil {
		return PointsResponse{}, err
	}
	defer projections.release()

	ctx, cancel := context.WithTimeout(parent, 5*time.Minute)
	defer cancel()

	cmd := exec.CommandContext(ctx, py, script, "--stdin", projectionMethod)
	cmd.Env = os.Environ()
	if qdrantKey := strings.TrimSpace(os.Getenv("QDRANT_API_KEY")); qdrantKey != "" {
		cmd.Env = append(cmd.Env, "QDRANT_API_KEY="+qdrantKey)
	}
	payload, err := json.Marshal(map[string]interface{}{"points": points, "vector_name": vectorName})
	if err != nil {
		return PointsResponse{}, fmt.Errorf("marshal projection payload: %w", err)
	}
	cmd.Stdin = bytes.NewReader(payload)

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return PointsResponse{}, fmt.Errorf("stdout pipe: %w", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return PointsResponse{}, fmt.Errorf("stderr pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return PointsResponse{}, fmt.Errorf("start projection worker: %w", err)
	}

	stderrDone := make(chan struct{})
	go func() {
		defer close(stderrDone)
		scanner := bufio.NewScanner(stderrPipe)
		for scanner.Scan() {
			log.Print(scanner.Text())
		}
	}()

	out, readErr := io.ReadAll(stdoutPipe)
	waitErr := cmd.Wait()
	<-stderrDone
	if readErr != nil {
		return PointsResponse{}, fmt.Errorf("read projection output: %w", readErr)
	}
	if waitErr != nil {
		return PointsResponse{}, fmt.Errorf("projection worker failed: %w", waitErr)
	}

	var resp PointsResponse
	if err := json.Unmarshal(out, &resp); err != nil {
		return PointsResponse{}, fmt.Errorf("projection worker bad JSON: %w — output: %.300s", err, string(out))
	}
	return resp, nil
}

// ── API responses ─────────────────────────────────────────────────────────────

const (
	errCodeBadRequest       = "bad_request"
	errCodeNotFound         = "not_found"
	errCodeMethodNotAllowed = "method_not_allowed"
	errCodeUnauthorized     = "unauthorized"
	errCodeInternal         = "internal"
	errCodeQdrant           = "qdrant_unreachable"
	errCodeProvider         = "provider_unavailable"
	errCodeProjection       = "projection_failed"
	errCodeSaturated        = "projection_saturated"
)

type apiErrorBody struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type apiErrorResponse struct {
	Error apiErrorBody `json:"error"`
}

func writeJSONValue(w http.ResponseWriter, status int, value interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(value); err != nil {
		log.Printf("write json response: %v", err)
	}
}

func writeJSONError(w http.ResponseWriter, status int, code, message string) {
	writeJSONValue(w, status, apiErrorResponse{Error: apiErrorBody{Code: code, Message: message}})
}

func writeServerError(w http.ResponseWriter, status int, code, message string, cause error) {
	if cause != nil {
		log.Printf("api error code=%s status=%d: %v", code, status, cause)
	}
	writeJSONError(w, status, code, message)
}

func writeProjectionError(w http.ResponseWriter, err error) {
	if errors.Is(err, errProjectionBusy) {
		writeJSONError(w, http.StatusTooManyRequests, errCodeSaturated, "too many projections in flight, retry shortly")
		return
	}
	if errors.Is(err, context.Canceled) {
		writeJSONError(w, http.StatusRequestTimeout, errCodeProjection, "projection cancelled")
		return
	}
	writeServerError(w, http.StatusInternalServerError, errCodeProjection, "projection failed", err)
}

// ── Auth ──────────────────────────────────────────────────────────────────────

const (
	authHeaderName  = "Authorization"
	tokenHeaderName = "X-VectorView-Token"
	tokenQueryName  = "token"
)

func newAuthMiddleware(token string, next http.Handler) http.Handler {
	token = strings.TrimSpace(token)
	if token == "" {
		return next
	}
	expected := []byte(token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/api/") {
			next.ServeHTTP(w, r)
			return
		}
		setCORS(w)
		if requestTokenMatches(r, expected) {
			next.ServeHTTP(w, r)
			return
		}
		w.Header().Set("WWW-Authenticate", `Bearer realm="vectorview"`)
		writeJSONError(w, http.StatusUnauthorized, errCodeUnauthorized, "missing or invalid access token")
	})
}

func requestTokenMatches(r *http.Request, expected []byte) bool {
	candidates := make([]string, 0, 3)
	if raw := strings.TrimSpace(r.Header.Get(authHeaderName)); raw != "" {
		if parts := strings.SplitN(raw, " ", 2); len(parts) == 2 && strings.EqualFold(parts[0], "bearer") {
			candidates = append(candidates, strings.TrimSpace(parts[1]))
		}
	}
	if raw := strings.TrimSpace(r.Header.Get(tokenHeaderName)); raw != "" {
		candidates = append(candidates, raw)
	}
	if raw := strings.TrimSpace(r.URL.Query().Get(tokenQueryName)); raw != "" {
		candidates = append(candidates, raw)
	}
	for _, candidate := range candidates {
		if subtle.ConstantTimeCompare([]byte(candidate), expected) == 1 {
			return true
		}
	}
	return false
}

func authState(token string) string {
	if strings.TrimSpace(token) == "" {
		return "disabled"
	}
	return "bearer token required on /api/*"
}

// ── Health ────────────────────────────────────────────────────────────────────

type healthQdrant struct {
	URL       string `json:"url"`
	Reachable bool   `json:"reachable"`
	LatencyMS int64  `json:"latency_ms"`
	Error     string `json:"error,omitempty"`
}

type healthProjection struct {
	InUse    int `json:"in_use"`
	Capacity int `json:"capacity"`
}

type healthResponse struct {
	Status        string           `json:"status"`
	UptimeSeconds int64            `json:"uptime_seconds"`
	Qdrant        healthQdrant     `json:"qdrant"`
	Caches        []string         `json:"caches"`
	Projections   healthProjection `json:"projections"`
}

func buildHealth(ctx context.Context, q *qdrantClient, caches []string, guard *projectionGuard, started time.Time) healthResponse {
	out := healthResponse{
		Status:        "ok",
		UptimeSeconds: int64(time.Since(started).Seconds()),
		Qdrant:        healthQdrant{URL: q.base},
		Caches:        caches,
		Projections:   healthProjection{InUse: guard.inUse(), Capacity: guard.capacity()},
	}
	if out.Caches == nil {
		out.Caches = []string{}
	}
	checkCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	begin := time.Now()
	_, err := q.listCollections(checkCtx)
	out.Qdrant.LatencyMS = time.Since(begin).Milliseconds()
	if err != nil {
		out.Status = "degraded"
		out.Qdrant.Error = err.Error()
		return out
	}
	out.Qdrant.Reachable = true
	return out
}

// ── HTTP handlers ─────────────────────────────────────────────────────────────

func main() {
	cfg := loadConfig()
	app, err := buildServer(cfg)
	if err != nil {
		log.Fatalf("bootstrap: %v", err)
	}

	streamCtx, streamCancel := context.WithCancel(context.Background())
	defer streamCancel()

	mux := app.routes()

	if cfg.metaBridgeEnabled {
		startMetaBridgeWatchers(streamCtx, app.q, app.hub, cfg.metaBridgeCollections, time.Duration(cfg.metaBridgePollSeconds)*time.Second)
	}
	if cfg.frontpocketEnabled {
		startFrontPocketWatchers(streamCtx, app.q, app.hub, cfg.frontpocketCollections, time.Duration(cfg.frontpocketPollSeconds)*time.Second)
	}
	if cfg.redisURL != "" {
		startRedisTelemetryBridge(streamCtx, cfg.redisURL, cfg.redisTelemetryChannel, app.hub)
	}

	addr := ":" + cfg.port
	log.Printf("🚀 VectorView running → http://localhost%s", addr)
	if warnings := logConfigReport(streamCtx, cfg, app); warnings > 0 && cfg.strictConfig {
		log.Fatalf("Strict config: %d warning(s) — refusing to start (unset VECTORVIEW_STRICT_CONFIG to continue)", warnings)
	}

	httpSrv := &http.Server{
		Addr:              addr,
		Handler:           newAuthMiddleware(cfg.authToken, mux),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	baseCtx, cancelBase := context.WithCancel(context.Background())
	defer cancelBase()
	httpSrv.BaseContext = func(net.Listener) context.Context { return baseCtx }

	shutdownCtx, stopSignals := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stopSignals()

	serveErr := make(chan error, 1)
	go func() {
		serveErr <- httpSrv.ListenAndServe()
	}()

	select {
	case err := <-serveErr:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("server error: %v", err)
		}
	case <-shutdownCtx.Done():
		log.Printf("Shutdown requested, draining connections…")
		cancelBase()
		drainCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if err := httpSrv.Shutdown(drainCtx); err != nil {
			log.Printf("Graceful shutdown incomplete: %v", err)
			_ = httpSrv.Close()
		}
	}

	streamCancel()
	log.Printf("VectorView stopped")
}

// ── Inline Go PCA for small search result sets ────────────────────────────────

func toFloatSlice(v interface{}) []float64 {
	switch t := v.(type) {
	case []float64:
		if len(t) == 0 {
			return nil
		}
		return t
	case []interface{}:
		if len(t) == 0 {
			return nil
		}
		out := make([]float64, 0, len(t))
		for _, elem := range t {
			n, ok := elem.(float64)
			if !ok {
				return nil
			}
			out = append(out, n)
		}
		return out
	default:
		return nil
	}
}

func vectorCandidates(v interface{}) [][]float64 {
	if vec := toFloatSlice(v); len(vec) > 0 {
		return [][]float64{vec}
	}
	obj, ok := v.(map[string]interface{})
	if !ok {
		return nil
	}

	keys := make([]string, 0, len(obj))
	for key := range obj {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	candidates := make([][]float64, 0, len(obj))
	appendCandidate := func(raw interface{}) {
		if vec := toFloatSlice(raw); len(vec) > 0 {
			candidates = append(candidates, vec)
			return
		}
		nested, ok := raw.(map[string]interface{})
		if !ok {
			return
		}
		if vec := toFloatSlice(nested["vector"]); len(vec) > 0 {
			candidates = append(candidates, vec)
		}
	}

	if raw, ok := obj["vector"]; ok {
		appendCandidate(raw)
	}
	for _, key := range keys {
		if key == "vector" {
			continue
		}
		appendCandidate(obj[key])
	}
	return candidates
}

func extractDenseVector(v interface{}) []float64 {
	candidates := vectorCandidates(v)
	if len(candidates) == 0 {
		return nil
	}
	return candidates[0]
}

func extractDenseVectorByDim(v interface{}, dim int) []float64 {
	if dim <= 0 {
		return nil
	}
	for _, vec := range vectorCandidates(v) {
		if len(vec) == dim {
			return vec
		}
	}
	return nil
}

func chooseProjectionDimension(points []QPoint) int {
	dimCounts := make(map[int]int)
	for _, p := range points {
		seen := make(map[int]struct{})
		for _, vec := range vectorCandidates(p.Vector) {
			dim := len(vec)
			if dim == 0 {
				continue
			}
			if _, exists := seen[dim]; exists {
				continue
			}
			seen[dim] = struct{}{}
			dimCounts[dim]++
		}
	}

	bestDim, bestCount := 0, 0
	for dim, count := range dimCounts {
		if count > bestCount || (count == bestCount && dim > bestDim) {
			bestDim = dim
			bestCount = count
		}
	}
	return bestDim
}

func buildPCAProjectionMeta(componentVariances []float64, totalVariance float64) *ProjectionMeta {
	axes := make([]ProjectionAxis, 0, 3)
	for i := 0; i < 3; i++ {
		variancePercent := 0.0
		if totalVariance > 1e-12 && i < len(componentVariances) {
			variancePercent = (componentVariances[i] / totalVariance) * 100.0
		}
		axes = append(axes, ProjectionAxis{
			Component:         fmt.Sprintf("PC%d", i+1),
			VarianceExplained: variancePercent,
		})
	}
	return &ProjectionMeta{Method: "pca", Axes: axes}
}

func goPCA3D(points []QPoint) ([]PointData, *ProjectionMeta) {
	type projectedPoint struct {
		raw    QPoint
		vector []float64
	}

	selectedDim := chooseProjectionDimension(points)
	if selectedDim == 0 {
		return nil, nil
	}

	valid := make([]projectedPoint, 0, len(points))
	for _, p := range points {
		vec := extractDenseVectorByDim(p.Vector, selectedDim)
		if len(vec) == 0 {
			continue
		}
		valid = append(valid, projectedPoint{raw: p, vector: vec})
	}

	n := len(valid)
	if n == 0 {
		return nil, nil
	}
	dim := selectedDim

	mean := make([]float64, dim)
	for _, p := range valid {
		for i, v := range p.vector {
			mean[i] += v
		}
	}
	for i := range mean {
		mean[i] /= float64(n)
	}

	centered := make([][]float64, n)
	for i, p := range valid {
		row := make([]float64, dim)
		for j, v := range p.vector {
			row[j] = v - mean[j]
		}
		centered[i] = row
	}

	comps := powerIteration(centered, n, dim, 3, 20)

	coords := make([][3]float64, n)
	for i, row := range centered {
		for c, comp := range comps {
			if c < 3 {
				coords[i][c] = dotProduct(row, comp)
			}
		}
	}

	totalVariance := 0.0
	if n > 1 {
		for _, row := range centered {
			for _, v := range row {
				totalVariance += v * v
			}
		}
		totalVariance /= float64(n - 1)
	}
	componentVariances := make([]float64, 3)
	if n > 1 {
		for axis := 0; axis < 3; axis++ {
			sumSq := 0.0
			for i := 0; i < n; i++ {
				value := coords[i][axis]
				sumSq += value * value
			}
			componentVariances[axis] = sumSq / float64(n-1)
		}
	}
	projectionMeta := buildPCAProjectionMeta(componentVariances, totalVariance)

	for axis := 0; axis < 3; axis++ {
		mn, mx := coords[0][axis], coords[0][axis]
		for i := 1; i < n; i++ {
			if coords[i][axis] < mn {
				mn = coords[i][axis]
			}
			if coords[i][axis] > mx {
				mx = coords[i][axis]
			}
		}
		r := mx - mn
		for i := range coords {
			if r < 1e-8 {
				coords[i][axis] = 0
			} else {
				coords[i][axis] = ((coords[i][axis]-mn)/r - 0.5) * 200.0
			}
		}
	}

	out := make([]PointData, n)
	for i, p := range valid {
		out[i] = PointData{
			ID:      p.raw.ID,
			X:       coords[i][0],
			Y:       coords[i][1],
			Z:       coords[i][2],
			Payload: p.raw.Payload,
		}
	}
	return out, projectionMeta
}

func powerIteration(data [][]float64, n, dim, k, iters int) [][]float64 {
	comps := make([][]float64, 0, k)
	deflated := make([][]float64, n)
	for i := range deflated {
		deflated[i] = append([]float64{}, data[i]...)
	}
	for c := 0; c < k; c++ {
		v := make([]float64, dim)
		v[c%dim] = 1.0
		for iter := 0; iter < iters; iter++ {
			tmp := make([]float64, n)
			for i, row := range deflated {
				tmp[i] = dotProduct(row, v)
			}
			newV := make([]float64, dim)
			for i, row := range deflated {
				for j := range newV {
					newV[j] += tmp[i] * row[j]
				}
			}
			norm := 0.0
			for _, x := range newV {
				norm += x * x
			}
			norm = goSqrt(norm)
			if norm < 1e-10 {
				break
			}
			for j := range newV {
				newV[j] /= norm
			}
			v = newV
		}
		comps = append(comps, v)
		for i, row := range deflated {
			proj := dotProduct(row, v)
			for j := range row {
				deflated[i][j] -= proj * v[j]
			}
		}
	}
	return comps
}

func dotProduct(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		if i < len(b) {
			s += a[i] * b[i]
		}
	}
	return s
}

func goSqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 50; i++ {
		z = (z + x/z) / 2
	}
	return z
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func setCORS(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Type", "application/json")
}
