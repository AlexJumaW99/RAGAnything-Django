# ingest_and_chat/urls.py
from django.urls import path
from . import views

app_name = "ingest_and_chat"

urlpatterns = [
    # Dashboard
    path("", views.dashboard, name="dashboard"),

    # Pipeline
    path("health/", views.health, name="health"),
    path("ingest/", views.ingest, name="ingest"),
    path("stop/", views.stop, name="stop"),

    # Sessions
    path("sessions/", views.sessions_list, name="sessions_list"),
    path("sessions/<uuid:session_id>/", views.session_detail, name="session_detail"),

    # Chat
    path("chat/", views.chat_send, name="chat"),

    # Conversations
    path("conversations/", views.conversations_list, name="conversations_list"),
    path("conversations/<uuid:conversation_id>/history/", views.conversation_history, name="conversation_history"),
    path("conversations/<uuid:conversation_id>/", views.conversation_delete, name="conversation_delete"),
]
