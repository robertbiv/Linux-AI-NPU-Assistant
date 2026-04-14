# SPDX-License-Identifier: GPL-3.0-or-later
"""Conversation history — in-memory storage with optional disk persistence.

Messages are kept in a plain Python list so they are always available for
context without any I/O.  The list is also written to a JSON file on disk
(in the user's data directory) so prior conversations survive restarts.

Design notes
------------
- No database dependency; JSON is self-contained and human-readable.
- Only text content is persisted.  Image attachments are *not* stored on
  disk (they can be large and are usually transient).  The ``has_image``
  flag lets the UI indicate that images were part of a turn.
- ``max_messages`` caps the in-memory list so RAM stays bounded during
  very long sessions.  Older messages are trimmed from the *front* (oldest
  first), preserving the most recent context.

Encryption
----------
Pass ``encrypt=True`` (the default when the ``cryptography`` package is
installed) to store the history file as a Fernet-encrypted blob.  The
symmetric key is kept in a separate ``history.key`` file in the same
directory, protected with ``0o600`` permissions.

::

    history = ConversationHistory(encrypt=True)   # key auto-created
    history.add("user", "Hello!")
    # ~/.local/share/linux-ai-npu-assistant/history.enc  ← ciphertext
    # ~/.local/share/linux-ai-npu-assistant/history.key  ← AES key (owner only)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from src.security import check_path_permissions, secure_write

logger = logging.getLogger(__name__)

_DEFAULT_MAX_MESSAGES = 200
_DEFAULT_HISTORY_DIR = (
    Path.home() / ".local" / "share" / "linux-ai-npu-assistant"
)
_DEFAULT_HISTORY_FILE = _DEFAULT_HISTORY_DIR / "history.json"
_DEFAULT_HISTORY_ENC  = _DEFAULT_HISTORY_DIR / "history.enc"
_DEFAULT_KEY_FILE     = _DEFAULT_HISTORY_DIR / "history.key"


# ── Encryption helpers ────────────────────────────────────────────────────────

def _fernet_available() -> bool:
    """Return True if the *cryptography* package is importable."""
    try:
        from cryptography.fernet import Fernet  # noqa: F401
        return True
    except ImportError:
        return False


def generate_encryption_key() -> bytes:
    """Generate a new Fernet key (32 bytes, URL-safe base64-encoded).

    Returns
    -------
    bytes
        A 44-byte URL-safe base64 string suitable for ``Fernet(key)``.
    """
    from cryptography.fernet import Fernet
    return Fernet.generate_key()


def load_or_create_key(key_path: Path) -> bytes:
    """Load an existing Fernet key from *key_path*, or create one.

    The key file is written with ``0o600`` permissions (owner read/write
    only).  If the file already exists and has correct permissions its
    contents are returned unchanged.

    Parameters
    ----------
    key_path:
        Path to the ``history.key`` file.

    Returns
    -------
    bytes
        The Fernet key bytes.
    """
    if key_path.exists():
        check_path_permissions(key_path, label="history key file")
        key = key_path.read_bytes().strip()
        if key:
            return key
        # Fall through to regenerate if file is empty / corrupt.

    key = generate_encryption_key()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically with 0o600 permissions.
    tmp = key_path.with_suffix(".key.tmp")
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(key)
        tmp.chmod(0o600)
        tmp.replace(key_path)
    except OSError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise

    logger.info("Generated new history encryption key: %s", key_path)
    return key


def encrypt_data(plaintext: str, key: bytes) -> str:
    """Encrypt *plaintext* with Fernet and return a base64 ciphertext string.

    The returned string is ASCII-safe and can be written to a text file.
    """
    from cryptography.fernet import Fernet
    f = Fernet(key)
    return f.encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_data(ciphertext: str, key: bytes) -> str:
    """Decrypt Fernet *ciphertext* and return the original plaintext.

    Raises
    ------
    cryptography.fernet.InvalidToken
        If the key is wrong or the data was tampered with.
    """
    from cryptography.fernet import Fernet
    f = Fernet(key)
    return f.decrypt(ciphertext.encode("ascii")).decode("utf-8")


@dataclass
class Message:
    """A single turn in the conversation."""

    role: str               # "user" | "assistant" | "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    has_image: bool = False  # True when the turn included a screenshot or uploaded image

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            role=d["role"],
            content=d["content"],
            timestamp=d.get("timestamp", ""),
            has_image=d.get("has_image", False),
        )


class ConversationHistory:
    """Thread-safe, persistent conversation history.

    Parameters
    ----------
    max_messages:
        Maximum number of messages kept in memory.  When the list exceeds
        this limit the oldest messages are removed first.
    persist_path:
        JSON file path for persistence.  Pass ``None`` to disable disk
        persistence (history lives only for the current session).  When
        *encrypt* is ``True`` the path is rewritten with a ``.enc``
        extension automatically.
    system_prompt:
        An optional system message prepended to every API call to establish
        the assistant's persona / instructions.
    encrypt:
        When ``True`` (and the ``cryptography`` package is installed), the
        history file is encrypted with Fernet symmetric encryption.  A key
        file is stored alongside the history file with ``0o600`` permissions.
        Defaults to ``True`` when *cryptography* is available, ``False``
        otherwise (graceful degradation).
    encryption_key:
        Optional pre-existing Fernet key bytes.  When omitted the key is
        loaded from (or created in) a ``history.key`` file next to the
        history file.
    """

    def __init__(
        self,
        max_messages: int = _DEFAULT_MAX_MESSAGES,
        persist_path: Path | str | None = _DEFAULT_HISTORY_FILE,
        system_prompt: str = "",
        encrypt: bool | None = None,          # None → auto-detect
        encryption_key: bytes | None = None,  # pre-supplied key (tests / custom setup)
    ) -> None:
        self._max = max_messages
        self._system_prompt = system_prompt
        self._messages: list[Message] = []
        self._lock = threading.Lock()

        # ── Encryption setup ────────────────────────────────────────────────
        # Auto-detect: enable if cryptography is importable and not disabled.
        if encrypt is None:
            encrypt = _fernet_available()
        self._encrypt = encrypt and _fernet_available()

        if self._encrypt:
            # Derive paths
            base = Path(persist_path) if persist_path else None
            if base is not None:
                # Store ciphertext in .enc sidecar next to the plain file.
                self._path: Path | None = base.with_suffix(".enc")
                key_path = base.parent / "history.key"
            else:
                self._path = None
                key_path = _DEFAULT_KEY_FILE

            if encryption_key is not None:
                self._key: bytes | None = encryption_key
            elif self._path is not None:
                try:
                    self._key = load_or_create_key(key_path)
                except OSError as exc:
                    logger.warning(
                        "Could not load/create history encryption key (%s); "
                        "disabling encryption for this session.",
                        exc,
                    )
                    self._encrypt = False
                    self._key = None
                    self._path = Path(persist_path) if persist_path else None
            else:
                self._key = None
        else:
            self._path = Path(persist_path) if persist_path else None
            self._key = None

        self._load()

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add(
        self,
        role: str,
        content: str,
        *,
        has_image: bool = False,
    ) -> Message:
        """Append a message and persist immediately.

        Parameters
        ----------
        role:
            ``"user"`` or ``"assistant"``.
        content:
            Text content of the message.
        has_image:
            Set to ``True`` when the turn included an image (screenshot or
            uploaded file).  The image itself is not stored here.

        Returns
        -------
        Message
            The newly added message object.
        """
        msg = Message(role=role, content=content, has_image=has_image)
        with self._lock:
            self._messages.append(msg)
            # Trim oldest messages if over the cap
            if len(self._messages) > self._max:
                self._messages = self._messages[-self._max :]
        self._save()
        return msg

    def clear(self) -> None:
        """Remove all messages from memory and erase the on-disk file."""
        with self._lock:
            self._messages.clear()
        self._save()
        logger.info("Conversation history cleared.")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def all_messages(self) -> list[Message]:
        """Return a snapshot of all messages (oldest first)."""
        with self._lock:
            return list(self._messages)

    def recent(self, n: int) -> list[Message]:
        """Return the *n* most recent messages."""
        with self._lock:
            return list(self._messages[-n:])

    def __iter__(self) -> Iterator[Message]:
        return iter(self.all_messages())

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)

    # ── API payload helpers ───────────────────────────────────────────────────

    def to_openai_messages(
        self,
        *,
        include_system: bool = True,
        max_context: int | None = None,
    ) -> list[dict]:
        """Return the message list in OpenAI ``/chat/completions`` format.

        Parameters
        ----------
        include_system:
            Prepend the system prompt if one is configured.
        max_context:
            Only include the most recent *max_context* messages (besides the
            system message).  Use this to avoid hitting context-length limits.
        """
        messages: list[dict] = []
        if include_system and self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        history = self.all_messages()
        if max_context is not None:
            history = history[-max_context:]

        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
        return messages

    def to_ollama_messages(
        self,
        *,
        max_context: int | None = None,
    ) -> list[dict]:
        """Return the message list in Ollama ``/api/chat`` format.

        Ollama's chat endpoint mirrors the OpenAI format, so this is a thin
        wrapper around :meth:`to_openai_messages`.
        """
        return self.to_openai_messages(
            include_system=True, max_context=max_context
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            with self._lock:
                data = [m.to_dict() for m in self._messages]
            # Atomic write with owner-only permissions (0o600) so conversation
            # history is never readable by other local users.
            secure_write(
                self._path,
                json.dumps(data, indent=2, ensure_ascii=False),
                mode=0o600,
            )
        except OSError as exc:
            logger.warning("Could not save conversation history: %s", exc)

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        # Warn if the history file is readable by group or world.
        check_path_permissions(self._path, label="conversation history file")
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            with self._lock:
                self._messages = [Message.from_dict(d) for d in raw]
                # Enforce max even on loaded history
                if len(self._messages) > self._max:
                    self._messages = self._messages[-self._max :]
            logger.info(
                "Loaded %d messages from %s", len(self._messages), self._path
            )
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Could not load conversation history: %s", exc)
