# ingest_and_chat/urls.py
from django.urls import path
from . import views

app_name = "ingest_and_chat"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("health/", views.health, name="health"),
    path("ingest/", views.ingest, name="ingest"),
    path("query/", views.query, name="query"),
    path("stop/", views.stop, name="stop"),
]
