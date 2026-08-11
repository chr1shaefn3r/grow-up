"""The date and age footer.

Age arithmetic is where this can be quietly wrong: a timelapse spans years, so a
month boundary handled naively is not a rounding error, it is a caption that
says the wrong thing on a face nobody can date by eye. The calendar cases here
are the ones that break the obvious implementation.

Everything down to TestDrawing needs nothing installed. The drawing tests are
gated on Pillow, which CI does not have.
"""

from __future__ import annotations

from datetime import date

import pytest

from grow_up import annotate

BIRTH = date(2020, 3, 14)


class TestMonthArithmetic:
    def test_the_day_before_a_birthday_is_still_the_month_before(self):
        assert annotate.months_between(BIRTH, date(2020, 4, 13)) == 0
        assert annotate.months_between(BIRTH, date(2020, 4, 14)) == 1

    def test_a_year_is_twelve_months(self):
        assert annotate.months_between(BIRTH, date(2021, 3, 14)) == 12

    def test_a_short_month_does_not_round_up(self):
        """Born on the 31st, not one month old on the 28th of February."""
        assert annotate.months_between(date(2021, 1, 31), date(2021, 2, 28)) == 0
        assert annotate.months_between(date(2021, 1, 31), date(2021, 3, 31)) == 2

    def test_a_leap_day_birth_counts_from_the_29th(self):
        assert annotate.months_between(date(2020, 2, 29), date(2021, 2, 28)) == 11
        assert annotate.months_between(date(2020, 2, 29), date(2021, 3, 1)) == 12


class TestAgeText:
    def test_days_are_counted_exactly(self):
        assert annotate.age_text(BIRTH, date(2020, 3, 15), "days") == "1 day"
        assert annotate.age_text(BIRTH, date(2020, 4, 14), "days") == "31 days"

    def test_large_day_counts_are_grouped(self):
        assert annotate.age_text(BIRTH, date(2023, 8, 27), "days") == "1,261 days"

    def test_months_are_whole_months(self):
        assert annotate.age_text(BIRTH, date(2023, 8, 27), "months") == "41 months"

    def test_year_months_reads_as_a_person_would_say_it(self):
        assert annotate.age_text(BIRTH, date(2023, 8, 27), "year_months") == \
            "3 years, 5 months"

    def test_a_whole_number_of_years_drops_the_months(self):
        assert annotate.age_text(BIRTH, date(2023, 3, 14), "year_months") == "3 years"

    def test_under_a_year_drops_the_years(self):
        assert annotate.age_text(BIRTH, date(2020, 8, 20), "year_months") == "5 months"

    def test_a_newborn_falls_back_to_days(self):
        """`0 months` is exactly wrong on the frames where change is fastest."""
        assert annotate.age_text(BIRTH, date(2020, 3, 20), "year_months") == "6 days"

    def test_the_birthday_itself_is_zero_days(self):
        assert annotate.age_text(BIRTH, BIRTH, "year_months") == "0 days"

    def test_a_photo_before_the_birth_date_says_nothing(self):
        """That means the birth date is wrong; '-8 days' would be worse."""
        assert annotate.age_text(BIRTH, date(2020, 3, 6), "days") == ""

    def test_off_says_nothing(self):
        assert annotate.age_text(BIRTH, date(2023, 1, 1), "off") == ""

    def test_an_unknown_granularity_names_the_valid_ones(self):
        with pytest.raises(ValueError, match="year_months"):
            annotate.age_text(BIRTH, date(2023, 1, 1), "fortnights")


class TestLanguages:
    AT_THREE_AND_FIVE = date(2023, 8, 27)

    @pytest.mark.parametrize("code,expected", [
        ("en", "3 years, 5 months"),
        ("de", "3 Jahre, 5 Monate"),
        ("fr", "3 ans, 5 mois"),
        ("es", "3 años y 5 meses"),
        ("it", "3 anni e 5 mesi"),
    ])
    def test_year_months_in_every_language(self, code, expected):
        assert annotate.age_text(BIRTH, self.AT_THREE_AND_FIVE, "year_months", code) \
            == expected

    @pytest.mark.parametrize("code,expected", [
        ("en", "1 day"), ("de", "1 Tag"), ("fr", "1 jour"),
        ("es", "1 día"), ("it", "1 giorno"),
    ])
    def test_the_singular_is_not_just_the_plural_without_an_s(self, code, expected):
        assert annotate.age_text(BIRTH, date(2020, 3, 15), "days", code) == expected

    def test_french_mois_is_invariable(self):
        """The reason plurals are a table and not a rule."""
        assert annotate.age_text(BIRTH, date(2020, 4, 14), "months", "fr") == "1 mois"
        assert annotate.age_text(BIRTH, date(2020, 6, 14), "months", "fr") == "3 mois"

    @pytest.mark.parametrize("code,expected", [
        ("en", "1,261 days"), ("de", "1.261 Tage"), ("fr", "1 261 jours"),
        ("es", "1.261 días"), ("it", "1.261 giorni"),
    ])
    def test_each_language_groups_thousands_its_own_way(self, code, expected):
        assert annotate.age_text(BIRTH, date(2023, 8, 27), "days", code) == expected

    def test_an_unknown_language_names_the_supported_ones(self):
        with pytest.raises(ValueError, match="de, en, es, fr, it"):
            annotate.age_text(BIRTH, date(2023, 1, 1), "days", "sv")


class TestDateFormatting:
    WHEN = date(2026, 8, 5)

    def test_the_default_is_iso(self):
        assert annotate.format_date(self.WHEN) == "2026-08-05"

    @pytest.mark.parametrize("pattern,expected", [
        ("DD.MM.YYYY", "05.08.2026"),
        ("MM/DD/YY", "08/05/26"),
        ("D MMMM YYYY", "5 August 2026"),
        ("MMM D, YYYY", "Aug 5, 2026"),
        ("YYYY", "2026"),
    ])
    def test_tokens(self, pattern, expected):
        assert annotate.format_date(self.WHEN, pattern) == expected

    def test_a_long_month_is_not_eaten_by_a_short_one(self):
        """MMMM must match before MM, or a month name comes out as '0808'."""
        assert annotate.format_date(self.WHEN, "MMMM") == "August"
        assert annotate.format_date(self.WHEN, "MMM") == "Aug"
        assert annotate.format_date(self.WHEN, "MM") == "08"

    @pytest.mark.parametrize("code,expected", [
        ("en", "5 August 2026"), ("de", "5. August 2026"), ("fr", "5 août 2026"),
        ("es", "5 agosto 2026"), ("it", "5 agosto 2026"),
    ])
    def test_month_names_are_localised(self, code, expected):
        pattern = "D. MMMM YYYY" if code == "de" else "D MMMM YYYY"
        assert annotate.format_date(self.WHEN, pattern, code) == expected

    def test_literal_text_survives(self):
        assert annotate.format_date(self.WHEN, "week of YYYY-MM-DD") == \
            "week of 2026-08-05"


class TestAnnotationSettings:
    def test_it_is_off_by_default(self):
        """An existing config has no [encode.annotate], and must not grow one."""
        assert annotate.Annotation.from_config(None).enabled is False
        assert annotate.Annotation.from_config({}).enabled is False

    def test_a_bad_language_is_caught_before_any_frame_is_drawn(self):
        with pytest.raises(ValueError, match="unknown language"):
            annotate.Annotation.from_config({"enabled": True, "language": "sv"})

    def test_a_bad_granularity_names_the_setting(self):
        with pytest.raises(ValueError, match="encode.annotate.age"):
            annotate.Annotation.from_config({"enabled": True, "age": "weeks"})

    def test_nothing_is_validated_while_disabled(self):
        """A half-written block should not block a run that ignores it anyway."""
        assert annotate.Annotation.from_config({"language": "sv"}).enabled is False

    def test_texts_puts_the_date_left_and_the_age_right(self):
        settings = annotate.Annotation.from_config({"enabled": True})
        assert settings.texts(date(2023, 8, 27), BIRTH) == \
            ("2023-08-27", "3 years, 5 months")

    def test_without_a_birth_date_only_the_date_is_rendered(self):
        settings = annotate.Annotation.from_config({"enabled": True})
        assert settings.texts(date(2023, 8, 27), None) == ("2023-08-27", "")

    def test_age_off_leaves_the_right_side_empty(self):
        settings = annotate.Annotation.from_config({"enabled": True, "age": "off"})
        assert settings.texts(date(2023, 8, 27), BIRTH) == ("2023-08-27", "")
        assert settings.wants_age is False

    def test_date_off_leaves_the_left_side_empty(self):
        settings = annotate.Annotation.from_config({"enabled": True, "date": False})
        assert settings.texts(date(2023, 8, 27), BIRTH)[0] == ""


class TestStoringTheBirthDate:
    """`encode` must never need the network, so the birth date is cached at index.

    The table arrives through CREATE TABLE IF NOT EXISTS rather than
    ADDED_COLUMNS, so a database from an earlier release picks it up with no
    migration to get wrong -- which is the point of checking an old one here.
    """

    @pytest.fixture()
    def conn(self, tmp_path):
        from grow_up import db
        return db.connect(tmp_path / "t.sqlite")

    def test_a_database_without_the_table_gains_it(self, tmp_path):
        import sqlite3

        from grow_up import db

        # assets as an earlier release created it, so the index in SCHEMA still
        # has the column it names.
        path = tmp_path / "old.sqlite"
        raw = sqlite3.connect(path)
        raw.execute(
            "CREATE TABLE assets (id TEXT PRIMARY KEY, local_datetime TEXT,"
            " file_created_at TEXT, updated_at TEXT, width INTEGER, height INTEGER,"
            " checksum TEXT, original_file_name TEXT, indexed_at TEXT NOT NULL)")
        raw.commit()
        raw.close()

        conn = db.connect(path)
        assert db.birth_date(conn) is None      # the table exists and is empty
        db.upsert_person(conn, "p1", "me", "Kid", "2020-03-14")
        assert db.birth_date(conn) == "2020-03-14"

    def test_no_birth_date_anywhere_is_none(self, conn):
        from grow_up import db

        db.upsert_person(conn, "p1", "me", "Kid", None)
        assert db.birth_date(conn) is None

    def test_the_account_that_filled_it_in_wins(self, conn):
        """Two accounts, one child: only one of them usually sets the field."""
        from grow_up import db

        db.upsert_person(conn, "p1", "me", "Kid", None)
        db.upsert_person(conn, "p2", "her", "Kid", "2020-03-14")
        assert db.birth_date(conn) == "2020-03-14"

    def test_a_later_index_picks_up_a_newly_filled_field(self, conn):
        from grow_up import db

        db.upsert_person(conn, "p1", "me", "Kid", None)
        db.upsert_person(conn, "p1", "me", "Kid", "2020-03-14")
        assert db.birth_date(conn) == "2020-03-14"


class TestDrawing:
    """Pillow is a runtime dependency but not a test one; CI installs three packages."""

    @pytest.fixture()
    def Image(self):
        return pytest.importorskip("PIL.Image", reason="needs Pillow")

    def frame(self, Image, colour):
        return Image.new("RGB", (400, 500), colour)

    def test_the_size_is_unchanged(self, Image):
        out = annotate.draw_footer(self.frame(Image, "white"), "left", "right")
        assert out.size == (400, 500)

    def test_the_picture_above_the_band_is_untouched(self, Image):
        source = self.frame(Image, (12, 34, 56))
        out = annotate.draw_footer(source, "2026-08-05", "3 years")

        band = round(500 * annotate.FONT_SCALE * annotate.BAND_SCALE)
        above = 500 - band - 1
        assert out.getpixel((200, above)) == (12, 34, 56)
        assert out.getpixel((200, 0)) == (12, 34, 56)

    @pytest.mark.parametrize("colour", ["white", "black", (255, 220, 0)])
    def test_the_band_darkens_any_background(self, colour):
        """Readability by construction: snow and a night shot both get a scrim."""
        Image = pytest.importorskip("PIL.Image")
        source = self.frame(Image, colour)
        out = annotate.draw_footer(source, "2026-08-05", "3 years")

        inside = out.getpixel((5, 495))
        assert sum(inside) < sum(source.getpixel((5, 495))) or colour == "black"
        assert sum(inside) < 3 * 200      # dark enough for white text

    def test_the_text_actually_lands_in_the_band(self, Image):
        plain = annotate.draw_footer(self.frame(Image, "black"))
        written = annotate.draw_footer(self.frame(Image, "black"), "2026-08-05", "3 years")
        assert plain.tobytes() != written.tobytes()

    def test_both_labels_are_drawn_on_their_own_side(self, Image):
        left_only = annotate.draw_footer(self.frame(Image, "black"), "2026-08-05", "")
        right_only = annotate.draw_footer(self.frame(Image, "black"), "", "3 years")

        band = round(500 * annotate.FONT_SCALE * annotate.BAND_SCALE)
        row = 500 - band // 2
        brightness = lambda img, x: sum(img.getpixel((x, row)))  # noqa: E731

        assert max(brightness(left_only, x) for x in range(10, 120)) > 300
        assert max(brightness(right_only, x) for x in range(280, 390)) > 300

    def test_a_configured_font_that_is_missing_is_an_error(self, Image, tmp_path):
        """Falling back would render the whole video in the wrong face."""
        with pytest.raises(RuntimeError, match="encode.annotate.font"):
            annotate.draw_footer(self.frame(Image, "black"), "x",
                                 configured_font=str(tmp_path / "nope.ttf"))
