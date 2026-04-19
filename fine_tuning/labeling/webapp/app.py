from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, flash, g, redirect, render_template, request, session, url_for

from .db import (
    assign_next_item,
    connect,
    fetch_admin_summary,
    fetch_existing_assignment,
    fetch_label_options,
    fetch_progress,
    get_or_create_annotator,
    initialize_database,
    load_label_options,
    recycle_item,
    submit_label,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _data_root() -> Path:
    return _repo_root() / "data"


def _label_options_path() -> Path:
    return _repo_root() / "fine_tuning" / "labeling" / "config" / "label_options.json"


def _workflow_config_path() -> Path:
    return _repo_root() / "fine_tuning" / "labeling" / "config" / "annotation_workflow.json"


def _db_path() -> Path:
    import json

    config = json.loads(_workflow_config_path().read_text(encoding="utf-8-sig"))
    return _repo_root() / config["database_path"]


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent / "templates"),
        static_folder=str(Path(__file__).resolve().parent / "static"),
    )
    app.secret_key = os.environ.get("ANNOTATION_SECRET_KEY", "dev-only-change-me")

    _data_root().joinpath("fine_tuning").mkdir(parents=True, exist_ok=True)

    with app.app_context():
        with connect(_db_path()) as connection:
            initialize_database(connection)
            load_label_options(connection, _label_options_path())

    def get_db():
        if "db" not in g:
            g.db = connect(_db_path())
        return g.db

    @app.teardown_appcontext
    def close_db(_exc):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    def current_annotator():
        annotator_id = session.get("annotator_id")
        if annotator_id is None:
            return None
        row = get_db().execute("SELECT * FROM annotators WHERE id = ?", (annotator_id,)).fetchone()
        if row is None:
            session.pop("annotator_id", None)
            session.pop("annotator_email", None)
        return row

    @app.route("/")
    def index():
        if session.get("annotator_id"):
            return redirect(url_for("annotate"))
        return redirect(url_for("register"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            if not email or "@" not in email:
                flash("Please enter a valid email address.")
                return redirect(url_for("register"))

            annotator = get_or_create_annotator(get_db(), email)
            session["annotator_id"] = annotator["id"]
            session["annotator_email"] = annotator["email"]
            flash(f"Registered as {annotator['email']}.")
            return redirect(url_for("annotate"))

        if session.get("annotator_id"):
            return redirect(url_for("annotate"))
        return render_template("register.html")

    @app.route("/logout")
    def logout():
        session.clear()
        flash("You have been signed out.")
        return redirect(url_for("register"))

    @app.route("/annotate", methods=["GET", "POST"])
    def annotate():
        annotator = current_annotator()
        if annotator is None:
            return redirect(url_for("register"))

        connection = get_db()

        if request.method == "POST":
            try:
                assignment_id = int(request.form["assignment_id"])
                primary_label = request.form["primary_label"]
                submit_label(
                    connection,
                    assignment_id=assignment_id,
                    annotator_id=annotator["id"],
                    primary_label=primary_label,
                )
            except Exception as exc:  # pragma: no cover - defensive UI guard
                flash(f"Could not save label: {exc}")
            return redirect(url_for("annotate"))

        assignment = fetch_existing_assignment(connection, annotator["id"])
        if assignment is None:
            assignment = assign_next_item(connection, annotator["id"])

        progress = fetch_progress(connection, annotator["id"])
        label_options = fetch_label_options(connection)
        if assignment is None:
            return render_template(
                "done.html",
                annotator=annotator,
                progress=progress,
            )

        return render_template(
            "annotate.html",
            annotator=annotator,
            assignment=assignment,
            label_options=label_options,
            progress=progress,
        )

    @app.route("/admin")
    def admin():
        summary = fetch_admin_summary(get_db())
        return render_template("admin.html", summary=summary)

    @app.route("/admin/recycle/<int:item_id>", methods=["POST"])
    def admin_recycle(item_id: int):
        recycle_item(get_db(), item_id)
        flash("Item moved into the recycle queue.")
        return redirect(url_for("admin"))

    @app.context_processor
    def inject_globals():
        return {"app_name": "SEC Risk Annotation"}

    return app
