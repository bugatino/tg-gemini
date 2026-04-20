"""Internationalization support for tg-gemini."""

from enum import StrEnum

__all__ = ["MESSAGES", "I18n", "Language", "MsgKey"]


class Language(StrEnum):
    """Supported languages."""

    EN = "en"
    ZH = "zh"


class MsgKey(StrEnum):
    """Message keys for translation lookups."""

    HELP = "help"
    SESSION_BUSY = "session_busy"
    SESSION_NEW = "session_new"
    MODEL_SWITCHED = "model_switched"
    MODEL_CURRENT = "model_current"
    MODE_SWITCHED = "mode_switched"
    MODE_CURRENT = "mode_current"
    STOP_OK = "stop_ok"
    ERROR_PREFIX = "error_prefix"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    UNKNOWN_CMD = "unknown_cmd"
    EMPTY_RESPONSE = "empty_response"
    SESSION_START_FAILED = "session_start_failed"
    # v2 additions
    LANG_SWITCHED = "lang_switched"
    LANG_CURRENT = "lang_current"
    QUIET_ON = "quiet_on"
    QUIET_OFF = "quiet_off"
    STATUS_INFO = "status_info"
    SESSION_LIST_HEADER = "session_list_header"
    SESSION_LIST_EMPTY = "session_list_empty"
    SESSION_SWITCHED = "session_switched"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_CURRENT = "session_current"
    SESSION_DELETED = "session_deleted"
    SESSION_DELETE_CONFIRM = "session_delete_confirm"
    SESSION_DELETE_CANCEL = "session_delete_cancel"
    SESSION_HISTORY_HEADER = "session_history_header"
    SESSION_HISTORY_EMPTY = "session_history_empty"
    SESSION_NAMED = "session_named"
    RATE_LIMITED = "rate_limited"
    PAGE_NAV = "page_nav"


MESSAGES: dict[MsgKey, dict[Language, str]] = {
    MsgKey.HELP: {
        Language.EN: (
            "Commands: /new – new session | /list – list sessions | /switch <target> – switch session"
            " | /delete – delete sessions | /name <new> – rename | /history – view history"
            " | /current – current session | /status – status info | /lang [code] – set language"
            " | /quiet – toggle quiet mode | /stop – stop agent | /model [name] – switch model"
            " | /mode [mode] – switch mode (default/auto_edit/yolo/plan) | /help – this help"
        ),
        Language.ZH: (
            "命令：/new – 新会话 | /list – 会话列表 | /switch <目标> – 切换会话"
            " | /delete – 删除会话 | /name <名称> – 重命名 | /history – 查看历史"
            " | /current – 当前会话 | /status – 状态信息 | /lang [语言] – 设置语言"
            " | /quiet – 切换静音模式 | /stop – 停止 agent | /model [名称] – 切换模型"
            " | /mode [模式] – 切换模式（default/auto_edit/yolo/plan）| /help – 帮助"
        ),
    },
    MsgKey.SESSION_BUSY: {
        Language.EN: "⏳ Agent is busy, please wait.",
        Language.ZH: "⏳ Agent 正忙，请稍候。",
    },
    MsgKey.SESSION_NEW: {
        Language.EN: "🆕 New session started.",
        Language.ZH: "🆕 已开始新会话。",
    },
    MsgKey.MODEL_SWITCHED: {
        Language.EN: "✅ Model switched to: {}",
        Language.ZH: "✅ 模型已切换为：{}",
    },
    MsgKey.MODEL_CURRENT: {
        Language.EN: "Current model: {}",
        Language.ZH: "当前模型：{}",
    },
    MsgKey.MODE_SWITCHED: {
        Language.EN: "✅ Mode switched to: {}",
        Language.ZH: "✅ 模式已切换为：{}",
    },
    MsgKey.MODE_CURRENT: {Language.EN: "Current mode: {}", Language.ZH: "当前模式：{}"},
    MsgKey.STOP_OK: {
        Language.EN: "🛑 Agent stopped.",
        Language.ZH: "🛑 Agent 已停止。",
    },
    MsgKey.ERROR_PREFIX: {Language.EN: "❌ Error: {}", Language.ZH: "❌ 错误：{}"},
    MsgKey.THINKING: {Language.EN: "💭 Thinking…", Language.ZH: "💭 思考中…"},
    MsgKey.TOOL_USE: {Language.EN: "🔧 {}: {}", Language.ZH: "🔧 {}：{}"},
    MsgKey.TOOL_RESULT: {Language.EN: "📋 Result: {}", Language.ZH: "📋 结果：{}"},
    MsgKey.UNKNOWN_CMD: {
        Language.EN: "Unknown command. Use /help for available commands.",
        Language.ZH: "未知命令，使用 /help 查看可用命令。",
    },
    MsgKey.EMPTY_RESPONSE: {Language.EN: "（no response）", Language.ZH: "（无响应）"},
    MsgKey.SESSION_START_FAILED: {
        Language.EN: "❌ Failed to start agent session: {}",
        Language.ZH: "❌ 启动 agent 会话失败：{}",
    },
    MsgKey.LANG_SWITCHED: {
        Language.EN: "✅ Language switched to: {}",
        Language.ZH: "✅ 语言已切换为：{}",
    },
    MsgKey.LANG_CURRENT: {
        Language.EN: "Current language: {}",
        Language.ZH: "当前语言：{}",
    },
    MsgKey.QUIET_ON: {
        Language.EN: "🔇 Quiet mode enabled.",
        Language.ZH: "🔇 静音模式已启用。",
    },
    MsgKey.QUIET_OFF: {
        Language.EN: "🔔 Quiet mode disabled.",
        Language.ZH: "🔔 静音模式已关闭。",
    },
    MsgKey.STATUS_INFO: {
        Language.EN: "Model: {}\nMode: {}\nSession: {}\nQuiet: {}",
        Language.ZH: "模型：{}\n模式：{}\n会话：{}\n静音：{}",
    },
    MsgKey.SESSION_LIST_HEADER: {
        Language.EN: "Sessions ({} total):",
        Language.ZH: "会话列表（共 {} 个）：",
    },
    MsgKey.SESSION_LIST_EMPTY: {
        Language.EN: "No sessions found.",
        Language.ZH: "暂无会话。",
    },
    MsgKey.SESSION_SWITCHED: {
        Language.EN: "✅ Switched to session: {}",
        Language.ZH: "✅ 已切换到会话：{}",
    },
    MsgKey.SESSION_NOT_FOUND: {
        Language.EN: "Session not found: {}",
        Language.ZH: "会话未找到：{}",
    },
    MsgKey.SESSION_CURRENT: {
        Language.EN: "Current session: {}",
        Language.ZH: "当前会话：{}",
    },
    MsgKey.SESSION_DELETED: {
        Language.EN: "{} session(s) deleted.",
        Language.ZH: "已删除 {} 个会话。",
    },
    MsgKey.SESSION_DELETE_CONFIRM: {
        Language.EN: "Delete {} session(s)?",
        Language.ZH: "确认删除 {} 个会话？",
    },
    MsgKey.SESSION_DELETE_CANCEL: {
        Language.EN: "Deletion cancelled.",
        Language.ZH: "已取消删除。",
    },
    MsgKey.SESSION_HISTORY_HEADER: {
        Language.EN: "Recent history:",
        Language.ZH: "近期历史记录：",
    },
    MsgKey.SESSION_HISTORY_EMPTY: {
        Language.EN: "No history.",
        Language.ZH: "暂无历史记录。",
    },
    MsgKey.SESSION_NAMED: {
        Language.EN: "✅ Session renamed to: {}",
        Language.ZH: "✅ 会话已重命名为：{}",
    },
    MsgKey.RATE_LIMITED: {
        Language.EN: "⏳ Rate limited. Try again later.",
        Language.ZH: "⏳ 请求频率超限，请稍后再试。",
    },
    MsgKey.PAGE_NAV: {Language.EN: "Page {} of {}", Language.ZH: "第 {} / {} 页"},
}


class I18n:
    """Internationalization helper for message translation."""

    def __init__(self, lang: Language | str = Language.EN):
        """Initialize with a language.

        Args:
            lang: Language enum or string code ("en" or "zh").
        """
        self._lang = Language(lang) if isinstance(lang, str) else lang

    @property
    def lang(self) -> Language:
        """Get the current language."""
        return self._lang

    def set_lang(self, lang: Language | str) -> None:
        """Set the current language.

        Args:
            lang: Language enum or string code ("en" or "zh").
        """
        self._lang = Language(lang) if isinstance(lang, str) else lang

    def t(self, key: MsgKey) -> str:
        """Translate a message key, falling back to EN.

        Args:
            key: The message key to translate.

        Returns:
            The translated string, or the key name if not found.
        """
        translations = MESSAGES.get(key, {})
        return translations.get(self._lang, translations.get(Language.EN, str(key)))

    def tf(self, key: MsgKey, *args: object) -> str:
        """Translate a message key and format with args.

        Args:
            key: The message key to translate.
            *args: Format arguments.

        Returns:
            The translated and formatted string.
        """
        return self.t(key).format(*args)

    @staticmethod
    def detect_language(text: str) -> Language:
        """Detect language from text content.

        CJK characters (Chinese, Japanese, Korean) are detected as ZH.
        Everything else defaults to EN.

        Args:
            text: The text to analyze.

        Returns:
            Detected Language enum.
        """
        for ch in text:
            cp = ord(ch)
            if (
                0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
                or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
                or 0x20000 <= cp <= 0x2A6DF  # CJK Extension B
            ):
                return Language.ZH
        return Language.EN
