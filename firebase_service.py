
import streamlit as st
import firebase_admin

from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import SERVER_TIMESTAMP


def get_firestore():
    """Initialize Firebase Firestore."""

    try:
        if not firebase_admin._apps:
            firebase_config = st.secrets.get("firebase", {})
            service_account = firebase_config.get(
                "service_account", {}
            )

            if not service_account:
                return None

            cred = credentials.Certificate(
                dict(service_account)
            )

            firebase_admin.initialize_app(cred)

        return firestore.client()

    except Exception as error:
        print(f"Firebase initialization error: {error}")
        return None


def load_favorites(db, visitor_id):
    """Load saved favorites."""

    if db is None:
        return []

    try:
        document = (
            db.collection("favorites")
            .document(visitor_id)
            .get()
        )

        if document.exists:
            return document.to_dict().get("items", [])

        return []

    except Exception as error:
        print(f"Loading favorites failed: {error}")
        return []


def set_favorite(db, visitor_id, item_id, is_favorite):
    """Add or remove an item from favorites."""

    if db is None:
        return False

    try:
        document_ref = (
            db.collection("favorites")
            .document(visitor_id)
        )

        document = document_ref.get()

        if document.exists:
            items = document.to_dict().get("items", [])
        else:
            items = []

        if is_favorite:
            if item_id not in items:
                items.append(item_id)
        else:
            items = [
                item for item in items
                if item != item_id
            ]

        document_ref.set(
            {
                "items": items,
                "updated_at": SERVER_TIMESTAMP,
            },
            merge=True,
        )

        return True

    except Exception as error:
        print(f"Saving favorite failed: {error}")
        return False


def save_search_event(
    db,
    visitor_id,
    query,
    source=None,
    country=None,
):
    """Save search information without uploading the image."""

    if db is None:
        return False

    try:
        db.collection("search_events").add(
            {
                "visitor_id": visitor_id,
                "query": str(query),
                "source": source,
                "country": country,
                "created_at": SERVER_TIMESTAMP,
            }
        )

        return True

    except Exception as error:
        print(f"Saving search event failed: {error}")

