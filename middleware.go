package main

import (
	"compress/gzip"
	"io"
	"io/fs"
	"log"
	"net/http"
	"strings"
	"sync"
)

var gzipWriterPool = sync.Pool{
	New: func() interface{} {
		writer, err := gzip.NewWriterLevel(io.Discard, gzip.DefaultCompression)
		if err != nil {
			return nil
		}
		return writer
	},
}

type gzipResponseWriter struct {
	http.ResponseWriter
	writer      *gzip.Writer
	wroteHeader bool
	passthrough bool
}

func (w *gzipResponseWriter) WriteHeader(status int) {
	if w.wroteHeader {
		return
	}
	w.wroteHeader = true
	switch status {
	case http.StatusNoContent, http.StatusNotModified:
		w.passthrough = true
		w.Header().Del("Content-Encoding")
	default:
		w.Header().Set("Content-Encoding", "gzip")
	}
	w.Header().Del("Content-Length")
	w.ResponseWriter.WriteHeader(status)
}

func (w *gzipResponseWriter) Write(data []byte) (int, error) {
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	if w.passthrough {
		return w.ResponseWriter.Write(data)
	}
	return w.writer.Write(data)
}

func (w *gzipResponseWriter) Flush() {
	if w.writer != nil {
		_ = w.writer.Flush()
	}
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func compressionMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("Vary", "Accept-Encoding")

		if !strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") ||
			r.Method == http.MethodHead ||
			r.Header.Get("Range") != "" ||
			r.URL.Path == "/api/stream" ||
			r.URL.Path == "/api/ws" {
			next.ServeHTTP(w, r)
			return
		}

		writer, ok := gzipWriterPool.Get().(*gzip.Writer)
		if !ok || writer == nil {
			next.ServeHTTP(w, r)
			return
		}
		writer.Reset(w)
		wrapped := &gzipResponseWriter{ResponseWriter: w, writer: writer}
		defer func() {
			_ = writer.Close()
			gzipWriterPool.Put(writer)
		}()
		next.ServeHTTP(wrapped, r)
	})
}

func staticFileHandler() http.Handler {
	staticSub, err := fs.Sub(staticFiles, "static")
	if err != nil {
		log.Fatalf("embed static assets: %v", err)
	}
	files := http.FileServer(http.FS(staticSub))

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/vendor/"), strings.HasPrefix(r.URL.Path, "/fonts/"):
			w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
		case strings.HasSuffix(r.URL.Path, ".png"), strings.HasSuffix(r.URL.Path, ".svg"):
			w.Header().Set("Cache-Control", "public, max-age=86400")
		default:
			w.Header().Set("Cache-Control", "no-cache")
		}
		files.ServeHTTP(w, r)
	})
}
