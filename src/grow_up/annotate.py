"""The date/age footer burned into the annotated video.

Everything except the drawing itself is plain Python: the age arithmetic, the
date formatting and the language tables carry all the behaviour worth testing,
and keeping Pillow out of them means they stay testable in a bare environment.
Same shape as `align.py`, where the transform maths is numpy and opencv appears
only for the warp.

Translations live here rather than coming from the stdlib `locale` module. That
module depends on locales being generated on the host, so the same config would
render differently on a Mac and on a Linux desktop -- which is exactly the pair
of machines this project runs on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

GRANULARITIES = ("off", "days", "months", "year_months")


@dataclass(frozen=True)
class Language:
    months: tuple[str, ...]        # 12 long names, January first
    short: tuple[str, ...]         # 12 abbreviations
    day: tuple[str, str]           # singular, plural
    month: tuple[str, str]
    year: tuple[str, str]
    group: str                     # thousands separator
    join: str                      # between the years and the months


LANGUAGES: dict[str, Language] = {
    "en": Language(
        months=("January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December"),
        short=("Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"),
        day=("day", "days"), month=("month", "months"), year=("year", "years"),
        group=",", join=", ",
    ),
    "de": Language(
        months=("Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
                "August", "September", "Oktober", "November", "Dezember"),
        short=("Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
               "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"),
        day=("Tag", "Tage"), month=("Monat", "Monate"), year=("Jahr", "Jahre"),
        group=".", join=", ",
    ),
    "fr": Language(
        months=("janvier", "février", "mars", "avril", "mai", "juin", "juillet",
                "août", "septembre", "octobre", "novembre", "décembre"),
        short=("janv.", "févr.", "mars", "avr.", "mai", "juin",
               "juil.", "août", "sept.", "oct.", "nov.", "déc."),
        # "mois" is invariable, which is the whole reason plurals are a table
        # rather than a rule with an s on the end.
        day=("jour", "jours"), month=("mois", "mois"), year=("an", "ans"),
        group=" ", join=", ",
    ),
    "es": Language(
        months=("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
                "agosto", "septiembre", "octubre", "noviembre", "diciembre"),
        short=("ene", "feb", "mar", "abr", "may", "jun",
               "jul", "ago", "sep", "oct", "nov", "dic"),
        day=("día", "días"), month=("mes", "meses"), year=("año", "años"),
        group=".", join=" y ",
    ),
    "it": Language(
        months=("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
                "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"),
        short=("gen", "feb", "mar", "apr", "mag", "giu",
               "lug", "ago", "set", "ott", "nov", "dic"),
        day=("giorno", "giorni"), month=("mese", "mesi"), year=("anno", "anni"),
        group=".", join=" e ",
    ),
}


def language(code: str) -> Language:
    try:
        return LANGUAGES[code]
    except KeyError:
        raise ValueError(
            f"unknown language {code!r}; supported: {', '.join(sorted(LANGUAGES))}"
        ) from None


# Longest first: the alternation is scanned left to right at each position, so
# MMMM has to be offered before MMM before MM or a month name becomes "0808".
TOKENS = ("YYYY", "YY", "MMMM", "MMM", "MM", "DD", "D")
_TOKEN_RE = re.compile("|".join(TOKENS))


def format_date(when: date, pattern: str = "YYYY-MM-DD", code: str = "en") -> str:
    """Render a date through a token pattern, with localised month names.

    Tokens rather than strftime directives because the config is written by a
    human: `YYYY-MM-DD` says what it does, `%Y-%m-%d` needs a manual page.
    """
    lang = language(code)
    replacements = {
        "YYYY": f"{when.year:04d}",
        "YY": f"{when.year % 100:02d}",
        "MMMM": lang.months[when.month - 1],
        "MMM": lang.short[when.month - 1],
        "MM": f"{when.month:02d}",
        "DD": f"{when.day:02d}",
        "D": str(when.day),
    }
    return _TOKEN_RE.sub(lambda m: replacements[m.group()], pattern)


def _grouped(value: int, lang: Language) -> str:
    return f"{value:,}".replace(",", lang.group) if lang.group else str(value)


def _count(value: int, unit: tuple[str, str], lang: Language) -> str:
    return f"{_grouped(value, lang)} {unit[0] if value == 1 else unit[1]}"


def months_between(birth: date, when: date) -> int:
    """Whole months elapsed.

    The day-of-month correction is what makes the 31st behave: born on 31
    January, someone is not one month old on 28 February.
    """
    months = (when.year - birth.year) * 12 + (when.month - birth.month)
    if when.day < birth.day:
        months -= 1
    return months


def age_text(birth: date, when: date, granularity: str = "year_months",
             code: str = "en") -> str:
    """The age to print, or "" when there is nothing sensible to say.

    A photo dated before the birth date means the birth date is wrong; printing
    "-3 days" would be a worse answer than printing nothing.
    """
    if granularity not in GRANULARITIES:
        raise ValueError(
            f"unknown age granularity {granularity!r}; "
            f"supported: {', '.join(GRANULARITIES)}"
        )
    if granularity == "off" or when < birth:
        return ""

    lang = language(code)
    if granularity == "days":
        return _count((when - birth).days, lang.day, lang)

    months = months_between(birth, when)
    if granularity == "months":
        return _count(months, lang.month, lang)

    years, rest = divmod(months, 12)
    if years and rest:
        return lang.join.join((_count(years, lang.year, lang),
                               _count(rest, lang.month, lang)))
    if years:
        return _count(years, lang.year, lang)
    if rest:
        return _count(rest, lang.month, lang)
    # Under a month old. "0 months" is exactly wrong on the frames where the
    # change is fastest, so fall through to the finer unit.
    return _count((when - birth).days, lang.day, lang)


@dataclass(frozen=True)
class Annotation:
    """The `[encode.annotate]` settings, validated once."""

    enabled: bool = False
    date: bool = True
    age: str = "year_months"
    language: str = "en"
    date_format: str = "YYYY-MM-DD"
    font: str = ""

    @classmethod
    def from_config(cls, section: dict | None) -> "Annotation":
        section = dict(section or {})
        settings = cls(
            enabled=bool(section.get("enabled", False)),
            date=bool(section.get("date", True)),
            age=str(section.get("age", "year_months")),
            language=str(section.get("language", "en")),
            date_format=str(section.get("date_format", "YYYY-MM-DD")),
            font=str(section.get("font", "")),
        )
        if settings.enabled:
            # Validate on the way in, so a typo is reported before a hundred
            # frames have been redrawn rather than on the first of them.
            language(settings.language)
            if settings.age not in GRANULARITIES:
                raise ValueError(
                    f"unknown encode.annotate.age {settings.age!r}; "
                    f"supported: {', '.join(GRANULARITIES)}"
                )
        return settings

    @property
    def wants_age(self) -> bool:
        return self.enabled and self.age != "off"

    def texts(self, when: date, birth: date | None) -> tuple[str, str]:
        """The (left, right) footer strings for one frame."""
        left = format_date(when, self.date_format, self.language) if self.date else ""
        right = ""
        if self.wants_age and birth is not None:
            right = age_text(birth, when, self.age, self.language)
        return left, right


# --------------------------------------------------------------------------- #
# Drawing. Pillow is imported inside these two, never at module scope.
# --------------------------------------------------------------------------- #

# Font size as a fraction of the frame height, then the band as a multiple of
# the font size. Relative so a 720p and a 4K render look the same.
FONT_SCALE = 0.042
BAND_SCALE = 1.9
MARGIN_SCALE = 0.035
SCRIM_ALPHA = 150          # out of 255, over black

SYSTEM_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)


def resolve_font(configured: str, size: int):
    """A scalable font, or an error saying which setting to fill in.

    A configured path that does not exist is an error rather than a fallback:
    silently ignoring it would render the whole video in the wrong face and look
    like the setting does not work.
    """
    from PIL import ImageFont

    if configured:
        path = Path(configured)
        if not path.exists():
            raise RuntimeError(f"encode.annotate.font: no such file: {path}")
        return ImageFont.truetype(str(path), size)

    for candidate in SYSTEM_FONTS:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow older than 10.1 has only the bitmap default
        raise RuntimeError(
            "no scalable font found; set encode.annotate.font to a .ttf or .otf"
        ) from None


def _footer_metrics(size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Type size, band height, side margin and stroke width for a frame size.

    Shared by the two ways the footer reaches a video -- composited here, or
    handed to ffmpeg as its own layer -- so the two cannot drift apart in
    appearance.
    """
    width, height = size
    type_size = max(10, round(height * FONT_SCALE))
    return (type_size, round(type_size * BAND_SCALE), round(width * MARGIN_SCALE),
            max(1, type_size // 14))


def footer_layer(size: tuple[int, int], left: str = "", right: str = "", *,
                 font=None, configured_font: str = ""):
    """The footer alone on transparent pixels, as an RGBA image.

    For the overlay path, where ffmpeg composites the footer *after* a
    transition has run. Baking it into the frames instead would put the date and
    age through the same dissolve as the photographs, so they would ghost
    between values -- and through `morph`, where motion estimation would warp
    the glyphs outright.
    """
    from PIL import Image, ImageDraw

    width, height = size
    type_size, band, margin, stroke = _footer_metrics(size)
    font = font or resolve_font(configured_font, type_size)

    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rectangle([(0, height - band), (width, height)], fill=(0, 0, 0, SCRIM_ALPHA))
    middle = height - band // 2
    if left:
        draw.text((margin, middle), left, font=font, anchor="lm",
                  fill=(255, 255, 255, 255), stroke_width=stroke,
                  stroke_fill=(0, 0, 0, 255))
    if right:
        draw.text((width - margin, middle), right, font=font, anchor="rm",
                  fill=(255, 255, 255, 255), stroke_width=stroke,
                  stroke_fill=(0, 0, 0, 255))
    return layer


def draw_footer(image, left: str = "", right: str = "", *, font=None,
                configured_font: str = ""):
    """Return a copy of `image` with a footer band and its two labels.

    The band is what makes this readable over anything: a photo can be white
    snow or a night shot, and outlined text alone is a gamble on both. The band
    darkens rather than replaces, so the picture still shows through it.

    Deliberately not expressed as `alpha_composite(image, footer_layer(...))`,
    tidy as that would be: the text is drawn straight onto the darkened photo
    here, so its antialiased edges blend against the picture. Against
    transparency they blend differently, and this is a shipped render.
    """
    from PIL import Image, ImageDraw

    width, height = image.size
    size, band, margin, stroke = _footer_metrics(image.size)
    font = font or resolve_font(configured_font, size)

    scrim = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(scrim).rectangle(
        [(0, height - band), (width, height)], fill=(0, 0, 0, SCRIM_ALPHA))
    out = Image.alpha_composite(image.convert("RGBA"), scrim)

    draw = ImageDraw.Draw(out)
    middle = height - band // 2
    if left:
        draw.text((margin, middle), left, font=font, anchor="lm",
                  fill=(255, 255, 255), stroke_width=stroke, stroke_fill=(0, 0, 0))
    if right:
        draw.text((width - margin, middle), right, font=font, anchor="rm",
                  fill=(255, 255, 255), stroke_width=stroke, stroke_fill=(0, 0, 0))
    return out.convert("RGB")
