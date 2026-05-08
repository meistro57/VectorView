package main

import (
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
					Size uint64 `json:"size"`
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
	Vector      interface{} `json:"vector"`
	Limit       int         `json:"limit"`
	WithPayload bool        `json:"with_payload"`
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
	SelectedID interface{}       `json:"selected_id"`
	Collection string            `json:"collection"`
	Limit      int               `json:"limit"`
	VectorName string            `json:"vector_name,omitempty"`
	Neighbors  []similarNeighbor `json:"neighbors"`
}

// ── API response types ────────────────────────────────────────────────────────

type PointsResponse struct {
	Points []PointData `json:"points"`
	Total  int         `json:"total"`
}

type PointData struct {
	ID      interface{}            `json:"id"`
	X       float64                `json:"x"`
	Y       float64                `json:"y"`
	Z       float64                `json:"z"`
	Payload map[string]interface{} `json:"payload"`
}

type CollectionMeta struct {
	Name        string `json:"name"`
	PointsCount uint64 `json:"points_count"`
	VectorSize  uint64 `json:"vector_size"`
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
		Name:        name,
		PointsCount: info.Result.PointsCount,
		VectorSize:  info.Result.Config.Params.Vectors.Size,
	}, nil
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
	if limit > 200 {
		limit = 200
	}
	return limit, nil
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

// runGPUPCA calls pca_gpu.py and returns projected PointData
func runGPUPCA(collection string, limit int, qdrantURL string) ([]PointData, error) {
	py := pythonBin()
	script := pcaScript()

	log.Printf("GPU PCA: %s %s %s %d %s", py, script, collection, limit, qdrantURL)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	cmd := exec.CommandContext(ctx, py, script,
		collection,
		strconv.Itoa(limit),
		qdrantURL,
	)
	cmd.Stderr = os.Stderr // stream [pca_gpu] logs to terminal

	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("pca_gpu.py failed: %w", err)
	}

	var resp PointsResponse
	if err := json.Unmarshal(out, &resp); err != nil {
		return nil, fmt.Errorf("pca_gpu.py bad JSON: %w — output: %.300s", err, string(out))
	}
	return resp.Points, nil
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
		metas := make([]CollectionMeta, 0)
		for _, name := range names {
			m, err := q.collectionMeta(r.Context(), name)
			if err == nil {
				metas = append(metas, m)
			}
		}
		json.NewEncoder(w).Encode(metas)
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

		log.Printf("→ /api/points collection=%s limit=%d", collection, limit)
		points, err := runGPUPCA(collection, limit, qdrantURL)
		if err != nil {
			log.Printf("ERROR runGPUPCA: %v", err)
			http.Error(w, err.Error(), 500)
			return
		}

		resp := PointsResponse{Points: points, Total: len(points)}
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

		sreq := searchReq{
			Filter: filterClause{Should: []filterMatch{
				{Key: "text", Match: matchValue{Text: queryStr}},
				{Key: "content", Match: matchValue{Text: queryStr}},
				{Key: "chunk_text", Match: matchValue{Text: queryStr}},
				{Key: "title", Match: matchValue{Text: queryStr}},
			}},
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

		// Re-project the filtered points via inline Go PCA
		// (search results are small enough — no need for GPU subprocess)
		var pts []QPoint
		if res.Result != nil {
			pts = res.Result.Points
		}
		projected := goPCA3D(pts)
		resp := PointsResponse{Points: projected, Total: len(projected)}
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

func extractDenseVector(v interface{}) []float64 {
	if vec := toFloatSlice(v); len(vec) > 0 {
		return vec
	}
	obj, ok := v.(map[string]interface{})
	if !ok {
		return nil
	}
	for _, raw := range obj {
		if vec := toFloatSlice(raw); len(vec) > 0 {
			return vec
		}
		nested, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		if vec := toFloatSlice(nested["vector"]); len(vec) > 0 {
			return vec
		}
	}
	return nil
}

func goPCA3D(points []QPoint) []PointData {
	type projectedPoint struct {
		raw    QPoint
		vector []float64
	}

	valid := make([]projectedPoint, 0, len(points))
	for _, p := range points {
		vec := extractDenseVector(p.Vector)
		if len(vec) == 0 {
			continue
		}
		valid = append(valid, projectedPoint{raw: p, vector: vec})
	}

	n := len(valid)
	if n == 0 {
		return nil
	}
	dim := len(valid[0].vector)

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
	return out
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
