"""
URL configuration for core project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path("api/rag/", include("ingest_and_chat.urls")),
    # Redirect root to the RAG dashboard
    path("", RedirectView.as_view(url="/api/rag/", permanent=False)),
]
