# ingest_and_chat/urls.py
from django.urls import path
from . import views

app_name = "ingest_and_chat"

urlpatterns = [
    # Dashboard (ingestion UI)
    path("", views.dashboard, name="dashboard"),

    # Chat Page UI
    path("chat-ui/", views.chat_page, name="chat_page"),

    # Pipeline
    path("health/", views.health, name="health"),
    path("ingest/", views.ingest, name="ingest"),
    path("stop/", views.stop, name="stop"),

    # LLM Providers
    path("providers/", views.providers_list, name="providers_list"),

    # Sessions
    path("sessions/", views.sessions_list, name="sessions_list"),
    path("sessions/<uuid:session_id>/", views.session_detail, name="session_detail"),

    # Chat API
    path("chat/", views.chat_send, name="chat"),

    # Conversations
    path("conversations/", views.conversations_list, name="conversations_list"),
    path("conversations/<uuid:conversation_id>/history/", views.conversation_history, name="conversation_history"),
    path("conversations/<uuid:conversation_id>/", views.conversation_delete, name="conversation_delete"),
]
