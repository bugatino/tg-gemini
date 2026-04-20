"""Card system for rich Telegram messages with interactive inline buttons."""

from dataclasses import dataclass, field
from enum import StrEnum

from tg_gemini.markdown import markdown_to_html

__all__ = [
    "ButtonStyle",
    "Card",
    "CardActions",
    "CardBuilder",
    "CardButton",
    "CardDivider",
    "CardElement",
    "CardHeader",
    "CardListItem",
    "CardMarkdown",
    "CardNote",
]

_DIVIDER = "──────────"


class ButtonStyle(StrEnum):
    PRIMARY = "primary"
    DEFAULT = "default"
    DANGER = "danger"


@dataclass
class CardButton:
    text: str
    callback_data: str
    style: ButtonStyle = ButtonStyle.DEFAULT


@dataclass
class CardMarkdown:
    text: str = ""


@dataclass
class CardDivider:
    pass


@dataclass
class CardActions:
    buttons: list[CardButton] = field(default_factory=list)


@dataclass
class CardNote:
    text: str = ""


@dataclass
class CardListItem:
    text: str = ""
    button: CardButton | None = None
    buttons: list[CardButton] = field(default_factory=list)


@dataclass
class CardHeader:
    title: str = ""
    color: str = ""  # display hint only — not rendered in Telegram


CardElement = CardMarkdown | CardDivider | CardActions | CardNote | CardListItem


@dataclass
class Card:
    header: CardHeader | None = None
    elements: list[CardElement] = field(default_factory=list)

    def render_text(self) -> str:
        """Render card content as Telegram HTML (without inline buttons)."""
        parts: list[str] = []
        if self.header and self.header.title:
            parts.append(f"<b>{self.header.title}</b>")
        for elem in self.elements:
            match elem:
                case CardMarkdown(text=t) if t:
                    parts.append(markdown_to_html(t))
                case CardDivider():
                    parts.append(_DIVIDER)
                case CardNote(text=t) if t:
                    parts.append(f"<i>{t}</i>")
                case CardListItem(text=t):
                    parts.append(f"• {t}")
                case CardActions():
                    pass  # buttons rendered separately via collect_buttons()
        return "\n".join(parts)

    def collect_buttons(self) -> list[list[CardButton]]:
        """Extract 2-D button list suitable for InlineKeyboardMarkup.

        CardActions → one row of buttons.
        CardListItem with button → one row with a single button.
        CardListItem with buttons → one row with all buttons.
        """
        rows: list[list[CardButton]] = []
        for elem in self.elements:
            match elem:
                case CardActions(buttons=btns) if btns:
                    rows.append(list(btns))
                case CardListItem(buttons=btns) if btns:
                    rows.append(list(btns))
                case CardListItem(button=btn) if btn is not None:
                    rows.append([btn])
        return rows

    def has_buttons(self) -> bool:
        return bool(self.collect_buttons())


class CardBuilder:
    """Fluent builder for Card objects.

    Example::

        card = (
            CardBuilder()
            .title("My Title", color="blue")
            .markdown("Hello **world**")
            .divider()
            .list_item("Session 1", CardButton("Switch", "act:cmd:/switch 1"))
            .actions(
                CardButton("OK", "act:confirm"), CardButton("Cancel", "act:cancel")
            )
            .note("Tip: use /help for commands")
            .build()
        )
    """

    def __init__(self) -> None:
        self._card = Card()

    def title(self, text: str, color: str = "") -> "CardBuilder":
        self._card.header = CardHeader(title=text, color=color)
        return self

    def markdown(self, text: str) -> "CardBuilder":
        self._card.elements.append(CardMarkdown(text=text))
        return self

    def divider(self) -> "CardBuilder":
        self._card.elements.append(CardDivider())
        return self

    def actions(self, *buttons: CardButton) -> "CardBuilder":
        self._card.elements.append(CardActions(buttons=list(buttons)))
        return self

    def note(self, text: str) -> "CardBuilder":
        self._card.elements.append(CardNote(text=text))
        return self

    def list_item(
        self,
        text: str,
        button: CardButton | None = None,
        *,
        buttons: list[CardButton] | None = None,
    ) -> "CardBuilder":
        self._card.elements.append(
            CardListItem(text=text, button=button, buttons=buttons or [])
        )
        return self

    def build(self) -> Card:
        return self._card
