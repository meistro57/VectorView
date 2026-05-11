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
	Points     []PointData     `json:"points"`
	Total      int             `json:"total"`
	Projection *ProjectionMeta `json:"projection,omitempty"`
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
	case "pca", "random", "tsne":
		return method, nil
	default:
		return "", fmt.Errorf("invalid projection: %q (expected pca, random, tsne)", raw)
	}
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

func pickSearchVector(raw interface{}) (interface{}, string) {
	if vec := toFloatSlice(raw); len(vec) > 0 {
		return vec, ""
	}
	obj, ok := raw.(map[string]interface{})
	if !ok {
		return nil, ""
	}

	keys := make([]string, 0, len(obj))
	for key := range obj {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	for _, key := range keys {
		if vec := toFloatSlice(obj[key]); len(vec) > 0 {
			if key == "vector" {
				return vec, ""
			}
			return namedSearchVector{Name: key, Vector: vec}, key
		}
		nested, ok := obj[key].(map[string]interface{})
		if !ok {
			continue
		}
		if vec := toFloatSlice(nested["vector"]); len(vec) > 0 {
			if key == "vector" {
				return vec, ""
			}
			return namedSearchVector{Name: key, Vector: vec}, key
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
func runGPUPCA(collection string, limit int, qdrantURL, projectionMethod, progressID string) (PointsResponse, error) {
	py := pythonBin()
	script := pcaScript()

	log.Printf("GPU PCA: %s %s %s %d %s", py, script, collection, limit, qdrantURL)
	setLoadProgress(progressID, loadProgress{Loaded: 0, Limit: limit, Percent: 0, Status: "starting", Done: false})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	cmd := exec.CommandContext(ctx, py, script,
		collection,
		strconv.Itoa(limit),
		qdrantURL,
		projectionMethod,
	)

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

func runProjectionOnPoints(points []QPoint, projectionMethod string) (PointsResponse, error) {
	py := pythonBin()
	script := pcaScript()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	cmd := exec.CommandContext(ctx, py, script, "--stdin", projectionMethod)
	payload, err := json.Marshal(map[string]interface{}{"points": points})
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

	staticSub, err := fs.Sub(staticFiles, "static")
	if err != nil {
		log.Fatal(err)
	}

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
	// Delegates fetch+PCA to pca_gpu.py
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
		progressID := strings.TrimSpace(r.URL.Query().Get("progress_id"))
		projectionMethod, err := parseProjectionMethod(r.URL.Query().Get("projection"))
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		log.Printf("→ /api/points collection=%s limit=%d projection=%s", collection, limit, projectionMethod)
		resp, err := runGPUPCA(collection, limit, qdrantURL, projectionMethod, progressID)
		if err != nil {
			setLoadProgress(progressID, loadProgress{Loaded: 0, Limit: limit, Percent: 0, Status: "failed", Done: true, Error: err.Error()})
			log.Printf("ERROR runGPUPCA: %v", err)
			http.Error(w, err.Error(), 500)
			return
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

		resp, err := runProjectionOnPoints(pts, projectionMethod)
		if err != nil {
			http.Error(w, fmt.Sprintf("projection failed: %v", err), http.StatusInternalServerError)
			return
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

		searchVector, vectorName := pickSearchVector(selected.Vector)
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

		searchVector, vectorName := pickSearchVector(selected.Vector)
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
