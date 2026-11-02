package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

type config struct {
	qdrantURL            string
	qdrantKey            string
	port                 string
	authToken            string
	maxPoints            int
	defaultCollection    string
	redisURL             string
	cacheDir             string
	cacheTTL             time.Duration
	cacheTTLSeconds      int
	cacheMaxMB           int64
	projectionSlots      int
	projectionQueue      time.Duration
	inlinePCAFallbackMax int
	strictConfig         bool
	python               string

	semanticProvider  string
	embedModel        string
	ollamaURL         string
	openrouterBaseURL string
	openrouterKey     string
	openrouterModel   string
	openaiBaseURL     string
	openaiKey         string
	openaiModel       string

	streamHeartbeatSeconds   int
	streamReplayDefault      int
	streamMaxEventsPerSecond int
	wsPingSeconds            int
	streamRecorderKey        string
	streamRecorderMaxLen     int64

	metaBridgeEnabled     bool
	metaBridgeCollections []string
	metaBridgePollSeconds int

	frontpocketEnabled     bool
	frontpocketCollections []string
	frontpocketPollSeconds int

	redisTelemetryChannel string
}

func loadConfig() config {
	loadEnvFiles()

	cfg := config{
		qdrantURL:                strings.TrimSpace(envOr("QDRANT_URL", "http://localhost:6333")),
		qdrantKey:                os.Getenv("QDRANT_API_KEY"),
		port:                     envOr("VECTORVIEW_PORT", "7433"),
		authToken:                strings.TrimSpace(os.Getenv("VECTORVIEW_TOKEN")),
		maxPoints:                envInt("VECTORVIEW_MAX_POINTS", 2000),
		defaultCollection:        envOr("VECTORVIEW_DEFAULT_COLLECTION", envOr("QDRANT_COLLECTION", defaultCollection)),
		redisURL:                 strings.TrimSpace(os.Getenv("VECTORVIEW_REDIS_URL")),
		cacheDir:                 envOr("VECTORVIEW_CACHE_DIR", filepath.Join(".cache", "projections")),
		cacheTTLSeconds:          envInt("VECTORVIEW_CACHE_TTL_SECONDS", 600),
		cacheMaxMB:               int64(envInt("VECTORVIEW_CACHE_MAX_MB", 512)),
		projectionSlots:          envInt("VECTORVIEW_MAX_PROJECTIONS", defaultProjectionSlots()),
		projectionQueue:          time.Duration(envInt("VECTORVIEW_PROJECTION_QUEUE_SECONDS", 30)) * time.Second,
		inlinePCAFallbackMax:     envInt("VECTORVIEW_INLINE_PCA_MAX", 2000),
		strictConfig:             envBool("VECTORVIEW_STRICT_CONFIG", false),
		python:                   envOr("VECTORVIEW_PYTHON", pythonBin()),
		semanticProvider:         strings.ToLower(strings.TrimSpace(envOr("VECTORVIEW_SEMANTIC_PROVIDER", "auto"))),
		embedModel:               strings.TrimSpace(envOr("VECTORVIEW_EMBED_MODEL", envOr("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))),
		ollamaURL:                strings.TrimSpace(envOr("VECTORVIEW_OLLAMA_URL", envOr("OLLAMA_BASE_URL", "http://localhost:11434"))),
		openrouterBaseURL:        strings.TrimSpace(envOr("VECTORVIEW_OPENROUTER_BASE_URL", envOr("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))),
		openrouterKey:            strings.TrimSpace(envOr("VECTORVIEW_OPENROUTER_API_KEY", os.Getenv("OPENROUTER_API_KEY"))),
		openrouterModel:          strings.TrimSpace(envOr("VECTORVIEW_OPENROUTER_MODEL", envOr("OPENROUTER_EMBEDDING_MODEL", "google/gemini-embedding-2-preview"))),
		openaiBaseURL:            strings.TrimSpace(envOr("VECTORVIEW_OPENAI_BASE_URL", envOr("OPENAI_BASE_URL", "https://api.openai.com/v1"))),
		openaiKey:                strings.TrimSpace(envOr("VECTORVIEW_OPENAI_API_KEY", os.Getenv("OPENAI_API_KEY"))),
		openaiModel:              strings.TrimSpace(envOr("VECTORVIEW_OPENAI_MODEL", envOr("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))),
		streamHeartbeatSeconds:   envInt("VECTORVIEW_STREAM_HEARTBEAT_SECONDS", 15),
		streamReplayDefault:      envInt("VECTORVIEW_STREAM_REPLAY_COUNT", 0),
		streamMaxEventsPerSecond: envInt("VECTORVIEW_STREAM_MAX_EVENTS_PER_SECOND", 30),
		wsPingSeconds:            envInt("VECTORVIEW_WS_PING_SECONDS", 20),
		streamRecorderKey:        strings.TrimSpace(envOr("VECTORVIEW_REDIS_STREAM_KEY", "vectorview.events")),
		streamRecorderMaxLen:     int64(envInt("VECTORVIEW_REDIS_STREAM_MAXLEN", 20000)),
		metaBridgeEnabled:        envBool("VECTORVIEW_META_BRIDGE_LIVE", true),
		metaBridgeCollections:    envList("VECTORVIEW_META_BRIDGE_COLLECTIONS", "mb_chunks,mb_claims"),
		metaBridgePollSeconds:    envInt("VECTORVIEW_META_BRIDGE_POLL_SECONDS", 3),
		frontpocketEnabled:       envBool("VECTORVIEW_FRONTPOCKET_LIVE", true),
		frontpocketCollections:   envList("VECTORVIEW_FRONTPOCKET_COLLECTIONS", "frontpocket_memory,minddrill_research_thoughts,minddrill_chat_memory,fp_reflections"),
		frontpocketPollSeconds:   envInt("VECTORVIEW_FRONTPOCKET_POLL_SECONDS", 3),
		redisTelemetryChannel:    strings.TrimSpace(envOr("VECTORVIEW_REDIS_PUBSUB_CHANNEL", "vectorview.telemetry")),
	}

	if cfg.maxPoints <= 0 {
		cfg.maxPoints = 2000
	}
	if cfg.cacheTTLSeconds <= 0 {
		cfg.cacheTTLSeconds = 600
	}
	if cfg.cacheMaxMB <= 0 {
		cfg.cacheMaxMB = 512
	}
	if cfg.projectionSlots < 1 {
		cfg.projectionSlots = 1
	}
	if cfg.projectionQueue <= 0 {
		cfg.projectionQueue = 30 * time.Second
	}
	if cfg.inlinePCAFallbackMax <= 0 {
		cfg.inlinePCAFallbackMax = 2000
	}
	if cfg.streamHeartbeatSeconds <= 0 {
		cfg.streamHeartbeatSeconds = 15
	}
	if cfg.streamReplayDefault < 0 {
		cfg.streamReplayDefault = 0
	}
	if cfg.streamMaxEventsPerSecond <= 0 {
		cfg.streamMaxEventsPerSecond = 30
	}
	if cfg.wsPingSeconds <= 0 {
		cfg.wsPingSeconds = 20
	}
	if cfg.streamRecorderMaxLen <= 0 {
		cfg.streamRecorderMaxLen = 20000
	}
	if cfg.metaBridgePollSeconds <= 0 {
		cfg.metaBridgePollSeconds = 3
	}
	if cfg.frontpocketPollSeconds <= 0 {
		cfg.frontpocketPollSeconds = 3
	}
	cfg.cacheTTL = time.Duration(cfg.cacheTTLSeconds) * time.Second
	defaultCollection = cfg.defaultCollection
	return cfg
}

func loadEnvFiles() {
	_ = godotenv.Load()
	if os.Getenv("OPENROUTER_API_KEY") != "" {
		return
	}
	for _, fallbackEnv := range []string{"../frontpocket/.env", "/home/mark/frontpocket/.env"} {
		if _, err := os.Stat(fallbackEnv); err != nil {
			continue
		}
		_ = godotenv.Load(fallbackEnv)
		if os.Getenv("OPENROUTER_API_KEY") != "" {
			return
		}
	}
}

func envInt(key string, def int) int {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return def
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return def
	}
	return value
}

func envBool(key string, def bool) bool {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return def
	}
	value, err := strconv.ParseBool(raw)
	if err != nil {
		return def
	}
	return value
}

func envList(key, def string) []string {
	raw := strings.TrimSpace(envOr(key, def))
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

type pythonProbe struct {
	path    string
	version string
	modules map[string]bool
	err     error
}

const pythonProbeScript = `import importlib.util, json, sys
mods = ["numpy", "umap", "sklearn", "torch", "cupy"]
out = {m: (importlib.util.find_spec(m) is not None) for m in mods}
out["python"] = sys.version.split()[0]
print(json.dumps(out))`

func probePython(ctx context.Context, python string) pythonProbe {
	if strings.TrimSpace(python) == "" {
		python = pythonBin()
	}
	probe := pythonProbe{path: python, modules: map[string]bool{}}
	cmd := exec.CommandContext(ctx, probe.path, "-c", pythonProbeScript)
	out, err := cmd.Output()
	if err != nil {
		probe.err = err
		return probe
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal(out, &decoded); err != nil {
		probe.err = err
		return probe
	}
	if version, ok := decoded["python"].(string); ok {
		probe.version = version
	}
	for name, present := range decoded {
		if flag, ok := present.(bool); ok {
			probe.modules[name] = flag
		}
	}
	return probe
}

func logConfigReport(ctx context.Context, cfg config, app *server) int {
	warnings := 0
	warn := func(format string, args ...interface{}) {
		warnings++
		log.Printf("   WARN "+format, args...)
	}

	checkCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	if _, err := app.q.listCollections(checkCtx); err != nil {
		warn("Qdrant unreachable at %s: %v (set QDRANT_URL / QDRANT_API_KEY)", cfg.qdrantURL, err)
	} else {
		log.Printf("   Qdrant: reachable at %s", cfg.qdrantURL)
	}

	switch {
	case cfg.redisURL == "":
		log.Printf("   Redis: not configured (disk cache only)")
	default:
		log.Printf("   Redis: configured at %s", cfg.redisURL)
	}
	log.Printf("   Cache: %s (ttl=%ds, max=%dMB)", strings.Join(app.cacheLabels, ", "), cfg.cacheTTLSeconds, cfg.cacheMaxMB)
	log.Printf("   Projection: %d slots, %ds queue wait, inline fallback <= %d points", app.projections.capacity(), int(app.projections.queueTimeout.Seconds()), app.inlinePCAFallbackMax)
	log.Printf("   Auth: %s", authState(cfg.authToken))

	probeCtx, probeCancel := context.WithTimeout(ctx, 10*time.Second)
	defer probeCancel()
	probe := probePython(probeCtx, app.python)
	if probe.err != nil {
		warn("projection worker unavailable: %s - %v (install python3 + numpy, or projections fall back to the inline Go PCA)", probe.path, probe.err)
	} else {
		log.Printf("   Projection worker: %s (python %s, numpy=%t)", probe.path, probe.version, probe.modules["numpy"])
		if !probe.modules["numpy"] {
			warn("numpy missing in %s - install it with `python3 -m pip install numpy`", probe.path)
		}
		for _, optional := range []struct{ module, feature string }{
			{"umap", "UMAP projection"},
			{"sklearn", "t-SNE projection"},
		} {
			if !probe.modules[optional.module] {
				log.Printf("   %s: unavailable (install %s)", optional.feature, optional.module)
			}
		}
	}

	ollamaCtx, ollamaCancel := context.WithTimeout(ctx, 1500*time.Millisecond)
	defer ollamaCancel()
	if err := pingOllama(ollamaCtx, cfg.ollamaURL); err != nil {
		log.Printf("   Ollama: unreachable at %s (%v)", cfg.ollamaURL, err)
	} else {
		log.Printf("   Ollama: reachable at %s (model %s)", cfg.ollamaURL, cfg.embedModel)
	}
	if cfg.openrouterKey == "" {
		warn("OPENROUTER_API_KEY missing - 3072-dim FrontPocket collections cannot be embedded")
	}
	if cfg.openaiKey == "" {
		log.Printf("   OpenAI: no key (1536-dim collections fall back to Ollama)")
	}
	return warnings
}

func pingOllama(ctx context.Context, base string) error {
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	if base == "" {
		return fmt.Errorf("ollama url not configured")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/api/tags", nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return fmt.Errorf("ollama %d", resp.StatusCode)
	}
	return nil
}
