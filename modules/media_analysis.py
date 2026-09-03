"""Reusable media analysis helpers for image and video attachments."""

import os
import tempfile
from collections import Counter
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from pocs.video_event_detection.video_diagnostics.preprocessing import VideoValidationError
from pocs.video_event_detection.video_diagnostics.service import VideoDiagnosticService


def draw_arrow(draw, start, end, color, width=6):
    draw.line([start, end], fill="#FFFFFF", width=width + 4)
    draw.line([start, end], fill=color, width=width)

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx**2 + dy**2) ** 0.5, 1)
    ux = dx / length
    uy = dy / length

    arrow_size = 18
    left = (
        end[0] - arrow_size * ux + arrow_size * 0.5 * uy,
        end[1] - arrow_size * uy - arrow_size * 0.5 * ux,
    )
    right = (
        end[0] - arrow_size * ux - arrow_size * 0.5 * uy,
        end[1] - arrow_size * uy + arrow_size * 0.5 * ux,
    )
    draw.polygon([end, left, right], fill="#FFFFFF")
    draw.polygon([end, left, right], fill=color)


def annotate_detections(image, result, names):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853"]

    for index, box in enumerate(result.boxes):
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = names[class_id]
        color = colors[index % len(colors)]
        outer_padding = 3

        draw.rectangle(
            [x1 - outer_padding, y1 - outer_padding, x2 + outer_padding, y2 + outer_padding],
            outline="#FFFFFF",
            width=8,
        )
        draw.rectangle([x1, y1, x2, y2], outline="#111111", width=5)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text = f"{label} {confidence:.2f}"
        text_box = draw.textbbox((0, 0), text)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]

        label_width = text_width + 18
        label_height = text_height + 14
        place_right = x2 < width * 0.65

        if place_right:
            label_x = min(max(x2 + 44, 16), max(width - label_width - 20, 16))
            arrow_end = (x2 + outer_padding, int((y1 + y2) / 2))
            anchor_x = label_x
        else:
            label_x = max(min(x1 - label_width - 44, width - label_width - 16), 16)
            arrow_end = (x1 - outer_padding, int((y1 + y2) / 2))
            anchor_x = label_x + label_width

        preferred_y = y1 - 12
        label_y = min(max(preferred_y, 12), max(height - label_height - 12, 12))
        arrow_start = (anchor_x, label_y + int(label_height / 2))

        draw_arrow(draw, arrow_start, arrow_end, color)
        draw.rounded_rectangle(
            [label_x, label_y, label_x + label_width, label_y + label_height],
            radius=12,
            fill="#FFFFFF",
            outline="#111111",
            width=4,
        )
        draw.rounded_rectangle(
            [label_x + 2, label_y + 2, label_x + label_width - 2, label_y + label_height - 2],
            radius=10,
            fill="#FFFFFF",
            outline=color,
            width=2,
        )
        draw.text((label_x + 9, label_y + 6), text, fill="#202124")

    return annotated


def build_detection_report(result, names):
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "label": names[class_id],
                "confidence": round(confidence, 4),
                "bbox": [x1, y1, x2, y2],
            }
        )

    return detections


def build_problem_statement(detections):
    if not detections:
        return "No defects were detected. If the vehicle still shows symptoms, inspect it manually and try a clearer image."

    counts = Counter(item["label"] for item in detections)
    labels = ", ".join(f"{label} ({count})" for label, count in counts.items())
    top_confidence = max(item["confidence"] for item in detections)
    return (
        f"Possible issues detected: {labels}. "
        f"Highest model confidence: {top_confidence:.2f}. "
        "These findings should be confirmed with a mechanic or vehicle inspector before repair decisions are made."
    )


def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, "Set the GEMINI_API_KEY environment variable to enable Gemini analysis."

    try:
        from google import genai
    except ImportError:
        return None, "Install the google-genai package to enable Gemini analysis."

    return genai.Client(api_key=api_key), None


def get_detection_crop(image, bbox, padding=18):
    x1, y1, x2, y2 = bbox
    width, height = image.size
    left = max(x1 - padding, 0)
    top = max(y1 - padding, 0)
    right = min(x2 + padding, width)
    bottom = min(y2 + padding, height)
    return image.crop((left, top, right, bottom))


def get_gemini_detection_detail(client, crop, label, confidence, related_messages):
    concern_text = (
        "User-reported messages:\n- " + "\n- ".join(related_messages)
        if related_messages
        else "User-reported messages: none provided"
    )

    prompt = (
        "You are assisting with vehicle image interpretation.\n"
        f"The object detector labeled this crop as '{label}' with confidence {confidence:.2f}.\n"
        f"{concern_text}\n\n"
        "Give a short, practical interpretation in 2-4 sentences.\n"
        "If the image suggests a more specific meaning inside the broad label, explain it cautiously.\n"
        "For example, if a dashboard indicator resembles a low fuel icon, say it may indicate low fuel.\n"
        "Do not claim certainty when the crop is ambiguous.\n"
        "Include a likely meaning and a next step."
    )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = temp_file.name
        crop.save(temp_path, format="PNG")

    try:
        uploaded = client.files.upload(file=temp_path)
        interaction = client.interactions.create(
            model="gemini-3.5-flash",
            input=[
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "uri": uploaded.uri,
                    "mime_type": uploaded.mime_type,
                },
            ],
        )
        return interaction.output_text.strip()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def get_gemini_video_analysis(client, attachment, related_messages, local_analysis):
    concern_text = (
        "User-reported messages:\n- " + "\n- ".join(related_messages)
        if related_messages
        else "User-reported messages: none provided"
    )

    prompt = (
        "You are assisting with vehicle video interpretation.\n"
        f"{concern_text}\n\n"
        "Local OpenCV findings (heuristic signals, not confirmed faults):\n"
        f"{local_analysis['summary']}\n"
        f"Events: {local_analysis['events']}\n\n"
        "Analyze this vehicle video and provide a practical summary in 4-6 sentences.\n"
        "Use the local findings as supporting context, and mention disagreements or uncertainty.\n"
        "Focus on visible defects, symptoms, indicators, or vehicle-condition clues.\n"
        "If the video is unclear, say so briefly and explain what better footage would help.\n"
        "End with a short recommended next step."
    )

    suffix = os.path.splitext(attachment["name"])[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(attachment["bytes"])

    try:
        uploaded = client.files.upload(file=temp_path)
        interaction = client.interactions.create(
            model="gemini-3.5-flash",
            input=[
                {"type": "text", "text": prompt},
                {
                    "type": "video",
                    "uri": uploaded.uri,
                    "mime_type": uploaded.mime_type,
                },
            ],
        )
        return interaction.output_text.strip()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def build_reports_context(reports, limit=4):
    if not reports:
        return "No prior inspection findings are available in this chat yet."

    lines = []
    for report in reports[-limit:]:
        if report["report_type"] == "video_analysis":
            local_summary = report.get("local_summary") or "No local summary is available."
            gemini_summary = report.get("video_summary") or "No optional Gemini summary is available."
            lines.append(f"- Video `{report['filename']}`: Local: {local_summary} Optional AI: {gemini_summary}")
            continue

        detections = report.get("detections", [])
        labels = ", ".join(dict.fromkeys(item["label"] for item in detections)) or "no defects detected"
        problem_statement = report.get("problem_statement") or "No problem statement available."
        lines.append(
            f"- Image `{report['filename']}`: labels found: {labels}. Summary: {problem_statement}"
        )

    return "\n".join(lines)


def build_message_context(messages, limit=8):
    recent_messages = []
    for message in messages[-limit:]:
        content = (message.get("content") or "").strip()
        if not content:
            continue
        role = "User" if message.get("role") == "user" else "Assistant"
        recent_messages.append(f"{role}: {content}")

    return "\n".join(recent_messages) if recent_messages else "No prior conversation yet."


def get_gemini_chat_reply(client, chat_session, user_message):
    conversation_context = build_message_context(chat_session.get("messages", []))
    reports_context = build_reports_context(chat_session.get("reports", []))
    prompt = (
        "You are a vehicle inspection assistant continuing an existing conversation.\n"
        "Use the conversation history and prior inspection findings to answer the latest user message.\n"
        "Stay grounded in what has already been detected or discussed. If something is uncertain, say so.\n"
        "Give practical, concise guidance in 3-6 sentences.\n"
        "If the user asks for next steps, prioritize the most important or safety-relevant issues first.\n\n"
        "Recent conversation:\n"
        f"{conversation_context}\n\n"
        "Prior inspection findings:\n"
        f"{reports_context}\n\n"
        "Latest user message:\n"
        f"{user_message}"
    )

    interaction = client.interactions.create(
        model="gemini-3.5-flash",
        input=[{"type": "text", "text": prompt}],
    )
    return interaction.output_text.strip()


def build_new_reports_context(reports):
    if not reports:
        return "No new attachment findings were created."

    lines = []
    for report in reports:
        if report["report_type"] == "video_analysis":
            local_summary = report.get("local_summary") or "No local summary is available."
            gemini_summary = report.get("video_summary") or "No optional Gemini summary is available."
            lines.append(f"- Video `{report['filename']}`: Local: {local_summary} Optional AI: {gemini_summary}")
            continue

        detections = report.get("detections", [])
        labels = ", ".join(dict.fromkeys(item["label"] for item in detections)) or "no defects detected"
        problem_statement = report.get("problem_statement") or "No problem statement available."
        lines.append(
            f"- Image `{report['filename']}`: labels found: {labels}. Summary: {problem_statement}"
        )

    return "\n".join(lines)


def get_gemini_post_analysis_reply(client, chat_session, user_message, new_reports):
    conversation_context = build_message_context(chat_session.get("messages", []))
    prior_reports_context = build_reports_context(chat_session.get("reports", []))
    new_reports_context = build_new_reports_context(new_reports)
    prompt = (
        "You are a vehicle inspection assistant continuing an ongoing conversation after new media was analyzed.\n"
        "Respond to the user's latest message using the new findings first, while keeping the prior conversation and "
        "earlier inspection results in mind.\n"
        "Be practical, concise, and conversational.\n"
        "Use 4-7 sentences.\n"
        "Mention the most relevant findings, what they may mean, and a sensible next step.\n"
        "If nothing clear was detected, say that plainly and suggest what better media or follow-up would help.\n\n"
        "Recent conversation:\n"
        f"{conversation_context}\n\n"
        "Prior inspection findings already in this chat:\n"
        f"{prior_reports_context}\n\n"
        "New attachment findings from the latest message:\n"
        f"{new_reports_context}\n\n"
        "Latest user message:\n"
        f"{user_message}"
    )

    interaction = client.interactions.create(
        model="gemini-3.5-flash",
        input=[{"type": "text", "text": prompt}],
    )
    return interaction.output_text.strip()


class AttachmentAnalysisService:
    """Handles YOLO and Gemini processing for chat attachments."""

    def __init__(
        self,
        model,
        gemini_cache,
        confidence_threshold,
        error_handler=None,
        video_service=None,
    ):
        self.model = model
        self.gemini_cache = gemini_cache
        self.error_handler = error_handler
        self.confidence_threshold = confidence_threshold
        self.video_service = video_service or VideoDiagnosticService()

    def analyze_attachments(self, current_chat, attachments_to_analyze):
        reports = []
        client, gemini_error = get_gemini_client()

        for attachment in attachments_to_analyze:
            related_messages = [attachment["message_text"]]
            attachment_kind = attachment.get("kind", "image")

            if attachment_kind == "video":
                video_report = self._analyze_video_attachment(
                    current_chat,
                    attachment,
                    related_messages,
                    client,
                    gemini_error,
                )
                if video_report:
                    reports.append(video_report)
                continue

            image_report = self._analyze_image_attachment(
                current_chat,
                attachment,
                related_messages,
                client,
                gemini_error,
            )
            if image_report:
                reports.append(image_report)

        assistant_summary = None
        if reports:
            if client:
                try:
                    latest_message = attachments_to_analyze[-1]["message_text"]
                    assistant_summary = get_gemini_post_analysis_reply(
                        client,
                        current_chat,
                        latest_message,
                        reports,
                    )
                except Exception as exc:
                    if self.error_handler:
                        self.error_handler(f"Gemini post-analysis reply is unavailable: {exc}")

            if not assistant_summary:
                summary_lines = []
                for report in reports:
                    if report["report_type"] == "video_analysis":
                        summary_lines.append(f"- {report['filename']}: {report['local_summary']}")
                    else:
                        labels = ", ".join(
                            dict.fromkeys(detection["label"] for detection in report["detections"])
                        ) or "no defects detected"
                        summary_lines.append(f"- {report['filename']}: {labels}")
                assistant_summary = "I analyzed the uploaded attachments.\n\n" + "\n".join(summary_lines)

        return reports, assistant_summary

    def generate_chat_reply(self, current_chat, user_message):
        client, gemini_error = get_gemini_client()
        if client:
            try:
                return get_gemini_chat_reply(client, current_chat, user_message)
            except Exception as exc:
                if self.error_handler:
                    self.error_handler(f"Gemini chat reply is unavailable: {exc}")

        if current_chat.get("reports"):
            return (
                "I saved your message and kept the earlier inspection context in this chat. "
                "Gemini chat response is unavailable right now, but you can continue using the detected issues below "
                "as the reference point for next steps."
            )

        return (
            "I saved your message. To give issue-specific guidance in this chat, attach an image or video first so I can "
            "ground the conversation in actual inspection findings."
        )

    def _analyze_video_attachment(self, current_chat, attachment, related_messages, client, gemini_error):
        video_summary = None
        video_error = gemini_error
        try:
            local_result = self.video_service.analyze(
                video_bytes=attachment["bytes"],
                filename=attachment["name"],
                user_description=related_messages[0] if related_messages else None,
            )
        except VideoValidationError as exc:
            if self.error_handler:
                self.error_handler(f"Could not process {attachment['name']}: {exc}")
            return None

        local_analysis = local_result.model_dump()

        if client:
            cache_key = (
                current_chat["id"],
                attachment["name"],
                tuple(related_messages),
                len(attachment["bytes"]),
            )
            if cache_key not in self.gemini_cache:
                try:
                    self.gemini_cache[cache_key] = get_gemini_video_analysis(
                        client,
                        attachment,
                        related_messages,
                        local_analysis,
                    )
                except Exception as exc:
                    self.gemini_cache[cache_key] = f"Video analysis is unavailable for this attachment: {exc}"
            video_summary = self.gemini_cache[cache_key]

        return {
            "report_type": "video_analysis",
            "filename": attachment["name"],
            "video_bytes": attachment["bytes"],
            "linked_messages": related_messages,
            "metadata": local_analysis["video"],
            "events": local_analysis["events"],
            "local_summary": local_analysis["summary"],
            "limitations": local_analysis["limitations"],
            "evidence": [item.model_dump() for item in local_result.to_evidence()],
            "video_summary": video_summary,
            "gemini_error": video_error,
        }

    def _analyze_image_attachment(self, current_chat, attachment, related_messages, client, gemini_error):
        try:
            image = Image.open(BytesIO(attachment["bytes"])).convert("RGB")
            image_np = np.array(image)
            results = self.model(image_np, conf=self.confidence_threshold)
        except Exception as exc:
            if self.error_handler:
                self.error_handler(f"Could not process {attachment['name']}: {exc}")
            return None

        result = results[0]
        detections = build_detection_report(result, self.model.names)
        problem_statement = build_problem_statement(detections)
        detailed_interpretations = []

        if client and detections:
            for detection in detections[:3]:
                cache_key = (
                    current_chat["id"],
                    attachment["name"],
                    detection["label"],
                    tuple(detection["bbox"]),
                    tuple(related_messages),
                )
                if cache_key not in self.gemini_cache:
                    crop = get_detection_crop(image, detection["bbox"])
                    try:
                        self.gemini_cache[cache_key] = get_gemini_detection_detail(
                            client,
                            crop,
                            detection["label"],
                            detection["confidence"],
                            related_messages,
                        )
                    except Exception as exc:
                        self.gemini_cache[cache_key] = (
                            f"Detailed AI interpretation is unavailable for this detection: {exc}"
                        )

                detailed_interpretations.append(
                    {
                        "label": detection["label"],
                        "detail": self.gemini_cache[cache_key],
                    }
                )

        return {
            "report_type": "image_detection",
            "filename": attachment["name"],
            "annotated_image": annotate_detections(image, result, self.model.names),
            "detections": detections,
            "problem_statement": problem_statement,
            "linked_messages": related_messages,
            "detailed_interpretations": detailed_interpretations,
            "gemini_error": gemini_error,
        }
