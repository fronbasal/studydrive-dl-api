# Studydrive Download API

This is a simple vibe-coded Studydrive PDF proxy inspired by [gookie-dev/studydrive](https://github.com/gookie-dev/studydrive). It exposes a single endpoint that enables download or viewing of Studydrive URLs. Also exposes a simple web UI.

The OpenAPI spec can be found at /docs. Deployment via compose has an optional Redis instance for caching PDF binary data.

Environment management is handled via Poetry.
