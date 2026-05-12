package main

import (
	"bufio"
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/joho/godotenv"
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

var pointsLoadProgress sync.Map

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

func (q *qdrantClient) getPointByID(ctx context.Context, collection, pointID string) (*QPoint, error) {
	idCandidates := []interface{}{pointID}
	if n, err := strconv.ParseInt(pointID, 10, 64); err == nil {
		idCandidates = append(idCandidates, n)
	}

	for _, idCandidate := range idCandidates {
		req := pointsByIDRequest{
			IDs:         []interface{}{idCandidate},
			WithPayload: true,
			WithVector:  true,
		}
		raw, status, err := q.do(ctx, http.MethodPost, "/collections/"+collection+"/points", req)
		if err != nil {
			return nil, err
		}
		if status >= 400 {
			return nil, fmt.Errorf("qdrant %d: %s", status, string(raw))
		}
		var res pointsByIDResult
		if err := json.Unmarshal(raw, &res); err != nil {
			return nil, fmt.Errorf("decode point lookup: %w", err)
		}
		if len(res.Result) > 0 {
			return &res.Result[0], nil
		}
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
		candidate := strings.TrimSuffix(exe, "/vectorview") + "/pca_gpu.py"
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

func setLoadProgress(progressID string, progress loadProgress) {
	if progressID == "" {
		return
	}
	pointsLoadProgress.Store(progressID, progress)
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

// runGPUPCA calls pca_gpu.py and returns projected points response
func runGPUPCA(collection string, limit int, qdrantURL, projectionMethod, vectorName, progressID string) (PointsResponse, error) {
	py := pythonBin()
	script := pcaScript()

	log.Printf("GPU PCA: %s %s %s %d %s", py, script, collection, limit, qdrantURL)
	setLoadProgress(progressID, loadProgress{Loaded: 0, Limit: limit, Percent: 0, Status: "starting", Done: false})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
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
				setLoadProgress(progressID, loadProgress{
					Loaded:  loaded,
					Limit:   progressLimit,
					Percent: clampPercent(loaded, progressLimit),
					Status:  "scrolling",
					Done:    false,
				})
				continue
			}
			if strings.Contains(line, "Fetch done") {
				setLoadProgress(progressID, loadProgress{Loaded: limit, Limit: limit, Percent: 100, Status: "projecting", Done: false})
			}
		}
	}()

	out, readErr := io.ReadAll(stdoutPipe)
	waitErr := cmd.Wait()
	<-stderrDone

	if readErr != nil {
		setLoadProgress(progressID, loadProgress{Loaded: 0, Limit: limit, Percent: 0, Status: "failed", Done: true, Error: readErr.Error()})
		return PointsResponse{}, fmt.Errorf("read pca_gpu.py output: %w", readErr)
	}
	if waitErr != nil {
		setLoadProgress(progressID, loadProgress{Loaded: 0, Limit: limit, Percent: 0, Status: "failed", Done: true, Error: waitErr.Error()})
		return PointsResponse{}, fmt.Errorf("pca_gpu.py failed: %w", waitErr)
	}

	var resp PointsResponse
	if err := json.Unmarshal(out, &resp); err != nil {
		setLoadProgress(progressID, loadProgress{Loaded: 0, Limit: limit, Percent: 0, Status: "failed", Done: true, Error: err.Error()})
		return PointsResponse{}, fmt.Errorf("pca_gpu.py bad JSON: %w — output: %.300s", err, string(out))
	}
	setLoadProgress(progressID, loadProgress{Loaded: len(resp.Points), Limit: limit, Percent: 100, Status: "done", Done: true})
	return resp, nil
}

func runProjectionOnPoints(points []QPoint, projectionMethod, vectorName string) (PointsResponse, error) {
	py := pythonBin()
	script := pcaScript()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	cmd := exec.CommandContext(ctx, py, script, "--stdin", projectionMethod)
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

// ── HTTP handlers ─────────────────────────────────────────────────────────────

func main() {
	_ = godotenv.Load()

	qdrantURL := envOr("QDRANT_URL", "http://localhost:6333")
	qdrantKey := os.Getenv("QDRANT_API_KEY")
	port := envOr("VECTORVIEW_PORT", "7433")
	maxPoints, _ := strconv.Atoi(envOr("VECTORVIEW_MAX_POINTS", "2000"))
	if maxPoints <= 0 {
		maxPoints = 2000
	}

	q := newQdrant(qdrantURL, qdrantKey)

	var pointsCache projectionCache
	if redisURL := strings.TrimSpace(os.Getenv("VECTORVIEW_REDIS_URL")); redisURL != "" {
		cacheTTLSeconds, err := strconv.Atoi(envOr("VECTORVIEW_CACHE_TTL_SECONDS", "600"))
		if err != nil || cacheTTLSeconds <= 0 {
			cacheTTLSeconds = 600
		}
		redisCache, err := newRedisProjectionCache(redisURL, time.Duration(cacheTTLSeconds)*time.Second)
		if err != nil {
			log.Printf("Projection cache disabled: %v", err)
		} else {
			pointsCache = redisCache
			log.Printf("Projection cache enabled via Redis (%s, ttl=%ds)", redisURL, cacheTTLSeconds)
		}
	}

	staticSub, err := fs.Sub(staticFiles, "static")
	if err != nil {
		log.Fatal(err)
	}

	semanticProvider := strings.ToLower(strings.TrimSpace(envOr("VECTORVIEW_SEMANTIC_PROVIDER", "ollama")))
	embedModel := strings.TrimSpace(envOr("VECTORVIEW_EMBED_MODEL", "nomic-embed-text"))
	ollamaURL := strings.TrimSpace(envOr("VECTORVIEW_OLLAMA_URL", "http://localhost:11434"))

	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.FS(staticSub)))

	// GET /api/collections
	mux.HandleFunc("/api/collections", func(w http.ResponseWriter, r *http.Request) {
		setCORS(w)
		names, err := q.listCollections(r.Context())
		if err != nil {
			log.Printf("ERROR listCollections: %v", err)
			http.Error(w, err.Error(), 500)
			return
		}
		metas := make([]CollectionMeta, 0, len(names))
		for _, name := range names {
			meta := CollectionMeta{Name: name}
			if m, err := q.collectionMeta(r.Context(), name); err == nil {
				meta = m
			} else {
				meta.ProjectionReady = false
				meta.ProjectionNote = "metadata unavailable"
			}
			ready, note := q.collectionProjectionStatus(r.Context(), name)
			meta.ProjectionReady = ready
			if note != "" {
				meta.ProjectionNote = note
			}
			metas = append(metas, meta)
		}
		json.NewEncoder(w).Encode(metas)
	})

	// GET /api/load-progress?id=progress_id
	mux.HandleFunc("/api/load-progress", func(w http.ResponseWriter, r *http.Request) {
		setCORS(w)
		id := strings.TrimSpace(r.URL.Query().Get("id"))
		if id == "" {
			http.Error(w, "id is required", http.StatusBadRequest)
			return
		}
		value, ok := pointsLoadProgress.Load(id)
		if !ok {
			http.Error(w, "progress id not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(value)
	})

	// GET /api/points?collection=X&limit=N
	// Delegates fetch+projection to pca_gpu.py and optionally serves Redis cache.
	mux.HandleFunc("/api/points", func(w http.ResponseWriter, r *http.Request) {
		setCORS(w)
		collection := r.URL.Query().Get("collection")
		if collection == "" {
			collection = "meistro_brain"
		}
		limit := maxPoints
		if ls := r.URL.Query().Get("limit"); ls != "" {
			if lv, err := strconv.Atoi(ls); err == nil && lv > 0 {
				limit = lv
			}
		}
		appendFrom, err := parseAppendFrom(r.URL.Query().Get("append_from"))
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if appendFrom > limit {
			http.Error(w, "append_from cannot exceed limit", http.StatusBadRequest)
			return
		}
		progressID := strings.TrimSpace(r.URL.Query().Get("progress_id"))
		vectorName := strings.TrimSpace(r.URL.Query().Get("vector_name"))
		projectionMethod, err := parseProjectionMethod(r.URL.Query().Get("projection"))
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		cacheKey := projectionCacheKey(collection, limit, projectionMethod, vectorName)

		if pointsCache != nil {
			if cached, ok, err := pointsCache.Get(r.Context(), cacheKey); err == nil && ok && cached != nil {
				if appendFrom > 0 {
					if appendFrom > len(cached.Points) {
						http.Error(w, "append_from exceeds cached total", http.StatusBadRequest)
						return
					}
					incremental := PointsResponse{
						Points:       append([]PointData(nil), cached.Points[appendFrom:]...),
						Total:        len(cached.Points),
						Projection:   cached.Projection,
						Incremental:  true,
						AppendFrom:   appendFrom,
						CachedResult: true,
					}
					json.NewEncoder(w).Encode(incremental)
					return
				}
				json.NewEncoder(w).Encode(cached)
				return
			} else if err != nil {
				log.Printf("projection cache read miss/error: %v", err)
			}
		}

		if appendFrom > 0 && projectionMethod == "random" && pointsCache != nil {
			baseKey := projectionCacheKey(collection, appendFrom, projectionMethod, vectorName)
			if base, ok, err := pointsCache.Get(r.Context(), baseKey); err == nil && ok && base != nil && len(base.Points) == appendFrom {
				deltaRaw, err := q.fetchPointsWindow(r.Context(), collection, appendFrom, limit-appendFrom)
				if err == nil {
					deltaResp, err := runProjectionOnPoints(deltaRaw, projectionMethod, vectorName)
					if err == nil {
						mergedPoints := make([]PointData, 0, len(base.Points)+len(deltaResp.Points))
						mergedPoints = append(mergedPoints, base.Points...)
						mergedPoints = append(mergedPoints, deltaResp.Points...)
						merged := &PointsResponse{Points: mergedPoints, Total: len(mergedPoints), Projection: deltaResp.Projection}
						if err := pointsCache.Set(r.Context(), cacheKey, merged); err != nil {
							log.Printf("projection cache write error: %v", err)
						}
						incremental := PointsResponse{
							Points:      append([]PointData(nil), deltaResp.Points...),
							Total:       len(mergedPoints),
							Projection:  deltaResp.Projection,
							Incremental: true,
							AppendFrom:  appendFrom,
						}
						json.NewEncoder(w).Encode(incremental)
						return
					}
					log.Printf("incremental projection fallback (worker): %v", err)
				} else {
					log.Printf("incremental projection fallback (fetch): %v", err)
				}
			}
		}

		log.Printf("→ /api/points collection=%s limit=%d projection=%s vector=%q", collection, limit, projectionMethod, vectorName)
		resp, err := runGPUPCA(collection, limit, qdrantURL, projectionMethod, vectorName, progressID)
		if err != nil {
			setLoadProgress(progressID, loadProgress{Loaded: 0, Limit: limit, Percent: 0, Status: "failed", Done: true, Error: err.Error()})
			log.Printf("ERROR runGPUPCA: %v", err)
			http.Error(w, err.Error(), 500)
			return
		}
		if pointsCache != nil {
			if err := pointsCache.Set(r.Context(), cacheKey, &resp); err != nil {
				log.Printf("projection cache write error: %v", err)
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	// GET /api/search?collection=X&q=term&limit=N
	// Payload keyword filter → GPU PCA on results
	mux.HandleFunc("/api/search", func(w http.ResponseWriter, r *http.Request) {
		setCORS(w)
		collection := r.URL.Query().Get("collection")
		if collection == "" {
			collection = "meistro_brain"
		}
		queryStr := r.URL.Query().Get("q")
		limit := 500
		if lv, err := strconv.Atoi(r.URL.Query().Get("limit")); err == nil && lv > 0 {
			limit = lv
		}
		projectionMethod, err := parseProjectionMethod(r.URL.Query().Get("projection"))
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		vectorName := strings.TrimSpace(r.URL.Query().Get("vector_name"))

		type matchValue struct {
			Text string `json:"text"`
		}
		type filterMatch struct {
			Key   string     `json:"key"`
			Match matchValue `json:"match"`
		}
		type filterClause struct {
			Should []filterMatch `json:"should"`
		}
		type searchReq struct {
			Filter      filterClause `json:"filter"`
			Limit       int          `json:"limit"`
			WithPayload bool         `json:"with_payload"`
			WithVectors bool         `json:"with_vectors"`
		}

		searchableKeys := []string{
			"text", "content", "chunk_text", "title", "summary", "claims", "concepts", "questions", "source_id", "file_source", "source_file", "tone",
			"TEXT", "CONTENT", "CHUNK_TEXT", "TITLE", "SUMMARY", "CLAIMS", "CONCEPTS", "QUESTIONS", "SOURCE_ID", "FILE_SOURCE", "SOURCE_FILE", "TONE",
		}
		matches := make([]filterMatch, 0, len(searchableKeys))
		for _, key := range searchableKeys {
			matches = append(matches, filterMatch{Key: key, Match: matchValue{Text: queryStr}})
		}

		sreq := searchReq{
			Filter:      filterClause{Should: matches},
			Limit:       limit,
			WithPayload: true,
			WithVectors: true,
		}

		raw, status, err := q.do(r.Context(), http.MethodPost,
			"/collections/"+collection+"/points/scroll", sreq)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		if status >= 400 {
			http.Error(w, fmt.Sprintf("qdrant %d: %s", status, string(raw)), 500)
			return
		}

		var res scrollResult
		if err := json.Unmarshal(raw, &res); err != nil {
			http.Error(w, fmt.Sprintf("decode scroll: %v", err), 500)
			return
		}

		var pts []QPoint
		if res.Result != nil {
			pts = res.Result.Points
		}

		resp, err := runProjectionOnPoints(pts, projectionMethod, vectorName)
		if err != nil {
			http.Error(w, fmt.Sprintf("projection failed: %v", err), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	// GET /api/semantic-search?collection=X&target_collection=Y&q=query&limit=N
	// Embed query text and run nearest-neighbor search in same or target collection.
	mux.HandleFunc("/api/semantic-search", func(w http.ResponseWriter, r *http.Request) {
		setCORS(w)
		sourceCollection := strings.TrimSpace(r.URL.Query().Get("collection"))
		if sourceCollection == "" {
			sourceCollection = "meistro_brain"
		}
		targetCollection := strings.TrimSpace(r.URL.Query().Get("target_collection"))
		if targetCollection == "" {
			targetCollection = sourceCollection
		}
		queryStr := strings.TrimSpace(r.URL.Query().Get("q"))
		if queryStr == "" {
			http.Error(w, "q is required", http.StatusBadRequest)
			return
		}
		limit := 120
		if lv, err := strconv.Atoi(r.URL.Query().Get("limit")); err == nil && lv > 0 {
			limit = lv
		}
		projectionMethod, err := parseProjectionMethod(r.URL.Query().Get("projection"))
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		vectorName := strings.TrimSpace(r.URL.Query().Get("vector_name"))

		var embedding []float64
		switch semanticProvider {
		case "ollama", "":
			embedding, err = embedQueryWithOllama(r.Context(), ollamaURL, embedModel, queryStr)
		default:
			http.Error(w, fmt.Sprintf("unsupported semantic provider: %s", semanticProvider), http.StatusBadRequest)
			return
		}
		if err != nil {
			http.Error(w, fmt.Sprintf("embedding failed: %v", err), http.StatusBadGateway)
			return
		}

		queryVector := interface{}(embedding)
		if vectorName != "" {
			queryVector = namedSearchVector{Name: vectorName, Vector: embedding}
		}
		sreq := searchPointsRequest{
			Vector:      queryVector,
			Limit:       limit,
			WithPayload: true,
			WithVectors: true,
		}
		raw, status, err := q.do(r.Context(), http.MethodPost,
			"/collections/"+targetCollection+"/points/search", sreq)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if status >= 400 {
			http.Error(w, fmt.Sprintf("qdrant %d: %s", status, string(raw)), http.StatusInternalServerError)
			return
		}

		var searchRes struct {
			Result []QPoint `json:"result"`
		}
		if err := json.Unmarshal(raw, &searchRes); err != nil {
			http.Error(w, fmt.Sprintf("decode semantic search: %v", err), http.StatusInternalServerError)
			return
		}
		resp, err := runProjectionOnPoints(searchRes.Result, projectionMethod, vectorName)
		if err != nil {
			http.Error(w, fmt.Sprintf("projection failed: %v", err), http.StatusInternalServerError)
			return
		}
		if sourceCollection != targetCollection {
			log.Printf("cross-collection semantic search: source=%s target=%s q=%q hits=%d", sourceCollection, targetCollection, queryStr, len(resp.Points))
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	// GET|POST /api/collections/{collection}/points/{point_id}/similar?limit=N
	mux.HandleFunc("/api/collections/{collection}/points/{point_id}/similar", func(w http.ResponseWriter, r *http.Request) {
		setCORS(w)
		if r.Method != http.MethodGet && r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		collection := strings.TrimSpace(r.PathValue("collection"))
		pointID := strings.TrimSpace(r.PathValue("point_id"))
		if collection == "" || pointID == "" {
			http.Error(w, "collection and point_id are required", http.StatusBadRequest)
			return
		}

		limit, err := parseSimilarityLimit(r, 12)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		selected, err := q.getPointByID(r.Context(), collection, pointID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if selected == nil {
			http.Error(w, "selected point not found", http.StatusNotFound)
			return
		}

		queryVectorName := strings.TrimSpace(r.URL.Query().Get("vector_name"))
		searchVector, vectorName := pickSearchVector(selected.Vector, queryVectorName)
		if searchVector == nil {
			http.Error(w, "selected point has no usable vector", http.StatusBadRequest)
			return
		}

		// Use the selected point vector directly as the Qdrant similarity query.
		sreq := searchPointsRequest{
			Vector:      searchVector,
			Limit:       limit + 1,
			WithPayload: true,
		}
		raw, status, err := q.do(r.Context(), http.MethodPost,
			"/collections/"+collection+"/points/search", sreq)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if status >= 400 {
			http.Error(w, fmt.Sprintf("qdrant %d: %s", status, string(raw)), http.StatusInternalServerError)
			return
		}

		var searchRes searchPointsResult
		if err := json.Unmarshal(raw, &searchRes); err != nil {
			http.Error(w, fmt.Sprintf("decode similar search: %v", err), http.StatusInternalServerError)
			return
		}

		selectedKey := canonicalPointID(selected.ID)
		neighbors := make([]similarNeighbor, 0, limit)
		for _, hit := range searchRes.Result {
			if canonicalPointID(hit.ID) == selectedKey {
				continue
			}
			neighbors = append(neighbors, similarNeighbor{ID: hit.ID, Score: hit.Score, Payload: hit.Payload})
			if len(neighbors) >= limit {
				break
			}
		}

		resp := similarResponse{
			SelectedID: selected.ID,
			Collection: collection,
			Limit:      limit,
			VectorName: vectorName,
			Neighbors:  neighbors,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	// GET|POST /api/collections/{collection}/points/{point_id}/similar-radius?radius=0.92&limit=400
	mux.HandleFunc("/api/collections/{collection}/points/{point_id}/similar-radius", func(w http.ResponseWriter, r *http.Request) {
		setCORS(w)
		if r.Method != http.MethodGet && r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		collection := strings.TrimSpace(r.PathValue("collection"))
		pointID := strings.TrimSpace(r.PathValue("point_id"))
		if collection == "" || pointID == "" {
			http.Error(w, "collection and point_id are required", http.StatusBadRequest)
			return
		}

		radius, err := parseSimilarityRadius(r, 0.92)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		limit, err := parseSimilarityLimit(r, 400)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		selected, err := q.getPointByID(r.Context(), collection, pointID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if selected == nil {
			http.Error(w, "selected point not found", http.StatusNotFound)
			return
		}

		queryVectorName := strings.TrimSpace(r.URL.Query().Get("vector_name"))
		searchVector, vectorName := pickSearchVector(selected.Vector, queryVectorName)
		if searchVector == nil {
			http.Error(w, "selected point has no usable vector", http.StatusBadRequest)
			return
		}

		sreq := searchPointsRequest{
			Vector:         searchVector,
			Limit:          limit + 1,
			WithPayload:    true,
			ScoreThreshold: &radius,
		}
		raw, status, err := q.do(r.Context(), http.MethodPost,
			"/collections/"+collection+"/points/search", sreq)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if status >= 400 {
			http.Error(w, fmt.Sprintf("qdrant %d: %s", status, string(raw)), http.StatusInternalServerError)
			return
		}

		var searchRes searchPointsResult
		if err := json.Unmarshal(raw, &searchRes); err != nil {
			http.Error(w, fmt.Sprintf("decode radius search: %v", err), http.StatusInternalServerError)
			return
		}

		selectedKey := canonicalPointID(selected.ID)
		neighbors := make([]similarNeighbor, 0, limit)
		for _, hit := range searchRes.Result {
			if canonicalPointID(hit.ID) == selectedKey {
				continue
			}
			neighbors = append(neighbors, similarNeighbor{ID: hit.ID, Score: hit.Score, Payload: hit.Payload})
			if len(neighbors) >= limit {
				break
			}
		}

		resp := similarResponse{
			SelectedID:       selected.ID,
			Collection:       collection,
			Limit:            limit,
			SimilarityRadius: &radius,
			VectorName:       vectorName,
			Neighbors:        neighbors,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	addr := ":" + port
	log.Printf("🚀 VectorView running → http://localhost%s", addr)
	log.Printf("   Qdrant: %s | GPU PCA script: %s | Max points: %d", qdrantURL, pcaScript(), maxPoints)
	log.Fatal(http.ListenAndServe(addr, mux))
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
