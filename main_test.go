package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCanonicalPointID(t *testing.T) {
	cases := []struct {
		input    interface{}
		expected string
	}{
		{"0000e554-7b69-5e18-840b-abe326eb94f1", "0000e554-7b69-5e18-840b-abe326eb94f1"},
		{int64(42), "42"},
		{float64(105), "105"},
		{uint64(999), "999"},
	}

	for _, c := range cases {
		actual := canonicalPointID(c.input)
		if actual != c.expected {
			t.Errorf("canonicalPointID(%v) = %q, want %q", c.input, actual, c.expected)
		}
	}
}

func TestChooseProjectionDimension(t *testing.T) {
	// Frontpocket vectors have 3072 dimensions
	vec3072 := make([]float64, 3072)
	points := []QPoint{
		{ID: "point-1", Vector: vec3072},
		{ID: "point-2", Vector: vec3072},
	}

	dim := chooseProjectionDimension(points)
	if dim != 3072 {
		t.Errorf("chooseProjectionDimension = %d, want 3072", dim)
	}
}

func TestPickSearchVector(t *testing.T) {
	vec3072 := make([]float64, 3072)
	for i := range vec3072 {
		vec3072[i] = 0.01
	}

	resVec, name := pickSearchVector(vec3072, "")
	if name != "" {
		t.Errorf("pickSearchVector name = %q, want empty", name)
	}
	slice, ok := resVec.([]float64)
	if !ok || len(slice) != 3072 {
		t.Fatalf("pickSearchVector slice len = %v, want 3072", len(slice))
	}
}

func TestEmbedQueryWithOpenRouter(t *testing.T) {
	expectedVector := []float64{0.1, 0.2, 0.3}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if r.Header.Get("X-Title") != "VectorView" {
			http.Error(w, "missing X-Title", http.StatusBadRequest)
			return
		}
		var req map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		if req["model"] != "google/gemini-embedding-2-preview" {
			http.Error(w, "unexpected model", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{
				{"embedding": expectedVector, "index": 0},
			},
		})
	}))
	defer server.Close()

	ctx := context.Background()
	emb, err := embedQueryWithOpenRouter(ctx, server.URL, "test-key", "google/gemini-embedding-2-preview", "test query", 3072)
	if err != nil {
		t.Fatalf("embedQueryWithOpenRouter failed: %v", err)
	}
	if len(emb) != len(expectedVector) {
		t.Errorf("got dim %d, want %d", len(emb), len(expectedVector))
	}
}

func TestEmbedQueryWithOpenAI(t *testing.T) {
	expectedVector := []float64{0.5, 0.6}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-openai-key" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{
				{"embedding": expectedVector, "index": 0},
			},
		})
	}))
	defer server.Close()

	ctx := context.Background()
	emb, err := embedQueryWithOpenAI(ctx, server.URL, "test-openai-key", "text-embedding-3-small", "test query", 1536)
	if err != nil {
		t.Fatalf("embedQueryWithOpenAI failed: %v", err)
	}
	if len(emb) != len(expectedVector) {
		t.Errorf("got dim %d, want %d", len(emb), len(expectedVector))
	}
}

func TestDetectCollectionEmbedding(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/collections/frontpocket_memory":
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"result": map[string]interface{}{
					"config": map[string]interface{}{
						"params": map[string]interface{}{
							"vectors": map[string]interface{}{
								"size": 3072,
							},
						},
					},
				},
			})
		case "/collections/frontpocket_memory/points/scroll":
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"result": map[string]interface{}{
					"points": []map[string]interface{}{
						{
							"id": "test-id",
							"payload": map[string]interface{}{
								"embedding_provider":   "openrouter",
								"embedding_model":      "google/gemini-embedding-2",
								"embedding_dimensions": 3072,
							},
						},
					},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	q := newQdrant(server.URL, "")
	ctx := context.Background()
	provider, model, dims := q.detectCollectionEmbedding(ctx, "frontpocket_memory")
	if provider != "openrouter" {
		t.Errorf("detectCollectionEmbedding provider = %q, want openrouter", provider)
	}
	if model != "google/gemini-embedding-2" {
		t.Errorf("detectCollectionEmbedding model = %q, want google/gemini-embedding-2", model)
	}
	if dims != 3072 {
		t.Errorf("detectCollectionEmbedding dims = %d, want 3072", dims)
	}
}
