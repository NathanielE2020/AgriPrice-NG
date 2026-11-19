"""Read-only Flask dashboard backend for AgriPrice-NG."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from src.wfp_nbs_dashboard_service import DashboardError, create_dashboard_service


def create_app(service=None) -> Flask:
    app = Flask(__name__)
    dashboard = service or create_dashboard_service()

    @app.get("/")
    def dashboard_page():
        return render_template("dashboard.html", deploy_scope=getattr(dashboard, "deploy_scope", ""))

    @app.get("/api/overview")
    def api_overview():
        return jsonify(dashboard.overview())

    @app.get("/api/series")
    def api_series():
        return jsonify(dashboard.series(request.args.get("dataset")))

    @app.get("/api/history")
    def api_history():
        dataset = request.args.get("dataset", "")
        series_id = request.args.get("series_id", "")
        return jsonify(
            dashboard.history(
                dataset,
                series_id,
                request.args.get("start_date"),
                request.args.get("end_date"),
            )
        )

    @app.get("/api/performance")
    def api_performance():
        dataset = request.args.get("dataset") or getattr(dashboard, "default_dataset", "wfp")
        series_id = request.args.get("series_id")
        return jsonify(dashboard.performance(dataset, series_id))

    @app.get("/api/forecast")
    def api_forecast():
        dataset = request.args.get("dataset", "")
        series_id = request.args.get("series_id", "")
        model = request.args.get("model", "auto")
        mode = request.args.get("mode", "")
        horizon_text = request.args.get("horizon", "")
        if not horizon_text:
            raise DashboardError("missing_horizon", "horizon is required and must be 1, 3, or 6.", 400)
        try:
            horizon = int(horizon_text)
        except ValueError as exc:
            raise DashboardError("invalid_horizon", "horizon must be 1, 3, or 6.", 400) from exc
        return jsonify(dashboard.forecast(dataset, series_id, model, horizon, mode=mode))

    @app.errorhandler(DashboardError)
    def handle_dashboard_error(error: DashboardError):
        return jsonify({"error": {"code": error.code, "message": error.message}}), error.status_code

    @app.errorhandler(404)
    def handle_not_found(error):  # noqa: ARG001 - Flask supplies the exception.
        if request.path.startswith("/api/"):
            return jsonify({"error": {"code": "not_found", "message": "The requested API endpoint was not found."}}), 404
        return "Page not found", 404

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):  # noqa: ARG001 - avoid exposing internals.
        if request.path.startswith("/api/"):
            return jsonify({"error": {"code": "internal_error", "message": "The dashboard could not complete the request."}}), 500
        return "The dashboard could not be loaded.", 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
