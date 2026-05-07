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
	Limit        int      `json:"limit"`
	WithPayload  bool     `json:"with_payload"`
	WithVectors  bool     `json:"with_vectors"`
	Offset       *uint64  `json:"offset,omitempty"`
}

type scrollResult struct {
	Result struct {
		Points    []QPoint `json:"points"`
		NextPage  *uint64  `json:"next_page_offset"`
	} `json:"result"`
}

type QPoint struct {
	ID      interface{}            `json:"id"`
	Vector  []float64              `json:"vector"`
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
	return &qdrantClient{base: base, key: key, http: &http.Client{Timeout: 60 * time.Second}}
}

func (q *qdrantClient) do(ctx context.Context, method, path string, body any) ([]byte, int, error) {
	var r io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
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
		return nil, err
	}
	names := make([]string, 0, len(res.Result.Collections))
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
		return CollectionMeta{}, err
	}
	return CollectionMeta{
		Name:        name,
		PointsCount: info.Result.PointsCount,
		VectorSize:  info.Result.Config.Params.Vectors.Size,
	}, nil
}

// scrollAll fetches up to maxPoints from the collection using pagination
func (q *qdrantClient) scrollAll(ctx context.Context, collection string, maxPoints int) ([]QPoint, error) {
	var all []QPoint
	var offset *uint64
	pageSize := 256
	if maxPoints < pageSize {
		pageSize = maxPoints
	}

	for {
		req := scrollRequest{
			Limit:       pageSize,
			WithPayload: true,
			WithVectors: true,
			Offset:      offset,
		}
		raw, _, err := q.do(ctx, http.MethodPost, "/collections/"+collection+"/points/scroll", req)
		if err != nil {
			return nil, fmt.Errorf("scroll: %w", err)
		}
		var res scrollResult
		if err := json.Unmarshal(raw, &res); err != nil {
			return nil, fmt.Errorf("scroll decode: %w", err)
		}
		all = append(all, res.Result.Points...)
		if res.Result.NextPage == nil || len(all) >= maxPoints {
			break
		}
		offset = res.Result.NextPage
	}

	if len(all) > maxPoints {
		all = all[:maxPoints]
	}
	return all, nil
}

// ── PCA: project N-dim vectors → 3D ──────────────────────────────────────────

func pca3D(points []QPoint) []PointData {
	n := len(points)
	if n == 0 {
		return nil
	}
	dim := len(points[0].Vector)
	if dim == 0 {
		return nil
	}

	// Center
	mean := make([]float64, dim)
	for _, p := range points {
		for i, v := range p.Vector {
			mean[i] += v
		}
	}
	for i := range mean {
		mean[i] /= float64(n)
	}

	centered := make([][]float64, n)
	for i, p := range points {
		row := make([]float64, dim)
		for j, v := range p.Vector {
			row[j] = v - mean[j]
		}
		centered[i] = row
	}

	// Power iteration for top 3 principal components
	comps := powerIteration(centered, n, dim, 3, 60)

	out := make([]PointData, n)
	for i, row := range centered {
		pd := PointData{
			ID:      points[i].ID,
			Payload: points[i].Payload,
		}
		if len(comps) > 0 {
			pd.X = dot(row, comps[0])
		}
		if len(comps) > 1 {
			pd.Y = dot(row, comps[1])
		}
		if len(comps) > 2 {
			pd.Z = dot(row, comps[2])
		}
		out[i] = pd
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
			// Av = data^T * (data * v)
			tmp := make([]float64, n)
			for i, row := range deflated {
				tmp[i] = dot(row, v)
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
			norm = sqrt64(norm)
			if norm < 1e-10 {
				break
			}
			for j := range newV {
				newV[j] /= norm
			}
			v = newV
		}
		comps = append(comps, v)

		// Deflate
		for i, row := range deflated {
			proj := dot(row, v)
			for j := range row {
				deflated[i][j] -= proj * v[j]
			}
		}
	}
	return comps
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		if i < len(b) {
			s += a[i] * b[i]
		}
	}
	return s
}

func sqrt64(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 50; i++ {
		z = (z + x/z) / 2
	}
	return z
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

	// Serve embedded static files
	staticSub, err := fs.Sub(staticFiles, "static")
	if err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.FS(staticSub)))

	// GET /api/collections → list all collections
	mux.HandleFunc("/api/collections", func(w http.ResponseWriter, r *http.Request) {
		setCORS(w)
		names, err := q.listCollections(r.Context())
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		metas := make([]CollectionMeta, 0, len(names))
		for _, name := range names {
			m, err := q.collectionMeta(r.Context(), name)
			if err == nil {
				metas = append(metas, m)
			}
		}
		json.NewEncoder(w).Encode(metas)
	})

	// GET /api/points?collection=meistro_brain&limit=2000
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

		raw, err := q.scrollAll(r.Context(), collection, limit)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}

		projected := pca3D(raw)
		resp := PointsResponse{Points: projected, Total: len(projected)}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	// GET /api/search?collection=meistro_brain&q=consciousness&limit=10
	mux.HandleFunc("/api/search", func(w http.ResponseWriter, r *http.Request) {
		setCORS(w)
		// Returns payload-only matches (text search via filter)
		collection := r.URL.Query().Get("collection")
		if collection == "" {
			collection = "meistro_brain"
		}
		queryStr := r.URL.Query().Get("q")
		limitStr := r.URL.Query().Get("limit")
		limit := 10
		if lv, err := strconv.Atoi(limitStr); err == nil && lv > 0 {
			limit = lv
		}

		// Use scroll with payload text filter (keyword match in text field)
		type filterMatch struct {
			Key   string `json:"key"`
			Match struct {
				Text string `json:"text"`
			} `json:"match"`
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

		fm := filterMatch{Key: "text"}
		fm.Match.Text = queryStr
		fm2 := filterMatch{Key: "content"}
		fm2.Match.Text = queryStr
		fm3 := filterMatch{Key: "chunk_text"}
		fm3.Match.Text = queryStr

		sreq := searchReq{
			Filter:      filterClause{Should: []filterMatch{fm, fm2, fm3}},
			Limit:       limit,
			WithPayload: true,
			WithVectors: true,
		}

		raw2, _, err := q.do(r.Context(), http.MethodPost,
			"/collections/"+collection+"/points/scroll", sreq)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		var res scrollResult
		json.Unmarshal(raw2, &res)
		projected := pca3D(res.Result.Points)
		resp := PointsResponse{Points: projected, Total: len(projected)}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	addr := ":" + port
	log.Printf("🚀 VectorView running → http://localhost%s", addr)
	log.Printf("   Qdrant: %s | Max points: %d", qdrantURL, maxPoints)
	log.Fatal(http.ListenAndServe(addr, mux))
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
