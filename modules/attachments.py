"""Helpers for working with chat attachments."""


def get_message_attachments(message):
    return message.get("metadata", {}).get("attachments", [])


def get_attachment_kind(attachment):
    mime_type = attachment.get("mime_type", "")
    if mime_type.startswith("video/"):
        return "video"
    return attachment.get("kind", "image")


def build_attachment_payload(uploaded_file):
    return {
        "name": uploaded_file.name,
        "bytes": uploaded_file.getvalue(),
        "mime_type": uploaded_file.type or "",
        "kind": "video" if (uploaded_file.type or "").startswith("video/") else "image",
    }


def build_analysis_attachment(uploaded_file, message_text):
    attachment = build_attachment_payload(uploaded_file)
    attachment["message_text"] = message_text
    return attachment
