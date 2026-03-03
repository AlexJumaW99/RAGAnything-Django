from django.apps import AppConfig


class IngestAndChatConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ingest_and_chat"
    verbose_name = "RAG Ingest & Chat"

    def ready(self):
        """
        Called once when Django starts.
        Heavy models (LLM, embeddings) are NOT loaded here —
        they are lazy-loaded on first use via config.get_llm(), etc.
        """
        pass
