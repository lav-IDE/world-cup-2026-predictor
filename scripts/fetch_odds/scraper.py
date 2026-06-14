import argparse
import logging
import re
import time
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# Logging — UTF-8 safe on Windows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scraper.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
import sys as _sys
if _sys.platform == "win32":
    import io as _io
    if hasattr(_sys.stdout, "buffer"):
        _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(_sys.stderr, "buffer"):
        _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer, encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

# Competition catalogue

BASE_URL = "https://www.oddsportal.com"

COMPETITIONS: dict[str, list[tuple[str, str]]] = {
    "FIFA World Cup": [
        ("world", "world-cup-2006"),
        ("world", "world-cup-2010"),
        ("world", "world-cup-2014"),
        ("world", "world-cup-2018"),
        ("world", "world-cup-2022"),
        ("world", "world-championship-2026"),
    ],
    "UEFA Euro": [
        ("europe", "euro-2008"),
        ("europe", "euro-2012"),
        ("europe", "euro-2016"),
        ("europe", "euro-2020"),
        ("europe", "euro-2024"),
    ],
    "UEFA Nations League": [
        ("europe", "uefa-nations-league-2018-2019"),
        ("europe", "uefa-nations-league-2020-2021"),
        ("europe", "uefa-nations-league-2022-2023"),
        ("europe", "uefa-nations-league-2024-2025"),
    ],
    "Copa America": [
        ("south-america", "copa-america-2007"),
        ("south-america", "copa-america-2011"),
        ("south-america", "copa-america-2015"),
        ("south-america", "copa-america-2016"),
        ("south-america", "copa-america-2019"),
        ("south-america", "copa-america-2021"),
        ("south-america", "copa-america"),
    ],
    "African Cup of Nations": [
        ("africa", "africa-cup-of-nations-2008"),
        ("africa", "africa-cup-of-nations-2010"),
        ("africa", "africa-cup-of-nations-2012"),
        ("africa", "africa-cup-of-nations-2013"),
        ("africa", "africa-cup-of-nations-2015"),
        ("africa", "africa-cup-of-nations-2017"),
        ("africa", "africa-cup-of-nations-2019"),
        ("africa", "africa-cup-of-nations-2021"),
        ("africa", "africa-cup-of-nations-2023"),
        ("africa", "africa-cup-of-nations-2025"),
    ],
    "AFC Asian Cup": [
        ("asia", "asian-cup-2011"),
        ("asia", "asian-cup-2015"),
        ("asia", "asian-cup-2019"),
        ("asia", "asian-cup-2023"),
    ],
    "Gold Cup": [
        ("north-central-america", "gold-cup-2009"),
        ("north-central-america", "gold-cup-2011"),
        ("north-central-america", "gold-cup-2013"),
        ("north-central-america", "gold-cup-2015"),
        ("north-central-america", "gold-cup-2017"),
        ("north-central-america", "gold-cup-2019"),
        ("north-central-america", "gold-cup-2021"),
        ("north-central-america", "gold-cup-2023"),
        ("north-central-america", "gold-cup"),
    ],
    "CONCACAF Nations League": [
        ("north-central-america", "concacaf-nations-league-2019-2020"),
        ("north-central-america", "concacaf-nations-league-2022-2023"),
        ("north-central-america", "concacaf-nations-league-2023-2024"),
        ("north-central-america", "concacaf-nations-league"),
    ],
    "Confederations Cup": [
        ("world", "fifa-confederations-cup-2009"),
        ("world", "fifa-confederations-cup-2013"),
        ("world", "fifa-confederations-cup"),
    ],
    "International Friendly": [
        ("world", "friendly-international-2008"),
        ("world", "friendly-international-2009"),
        ("world", "friendly-international-2010"),
        ("world", "friendly-international-2011"),
        ("world", "friendly-international-2012"),
        ("world", "friendly-international-2013"),
        ("world", "friendly-international-2014"),
        ("world", "friendly-international-2015"),
        ("world", "friendly-international-2016"),
        ("world", "friendly-international-2017"),
        ("world", "friendly-international-2018"),
        ("world", "friendly-international-2019"),
        ("world", "friendly-international-2020"),
        ("world", "friendly-international-2021"),
        ("world", "friendly-international-2022"),
        ("world", "friendly-international-2023"),
        ("world", "friendly-international-2024"),
        ("world", "friendly-international-2025"),
        ("world", "friendly-international"),
    ],
}

ALWAYS_NEUTRAL = {
    "FIFA World Cup", "UEFA Euro", "Copa America", "African Cup of Nations",
    "AFC Asian Cup", "Gold Cup", "Confederations Cup",
}

TARGET_TEAMS = {
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia & Herzegovina", "Bosnia and Herzegovina", "Brazil", "Canada", "Cape Verde",
    "Colombia", "Croatia", "Curacao", "Curaçao", "Czech Republic", "DR Congo", "D.R. Congo",
    "Ecuador", "Egypt", "England", "France", "Germany", "Ghana",
    "Haiti", "Iran", "Iraq", "Ivory Coast", "Japan", "Jordan",
    "Mexico", "Morocco", "Netherlands", "New Zealand", "Norway",
    "Panama", "Paraguay", "Portugal", "Qatar", "Saudi Arabia",
    "Scotland", "Senegal", "South Africa", "South Korea", "Spain",
    "Sweden", "Switzerland", "Tunisia", "Turkey", "United States", "USA",
    "Uruguay", "Uzbekistan",
}

QUAL_RE = re.compile(r"qualification", re.IGNORECASE)

# Scraper

class OddsPortalScraper:
    def __init__(self, headless: bool = True, delay: tuple = (2.0, 4.0)):
        self.headless = headless
        self.delay = delay
        self._pw = self._browser = self._ctx = self._page = None

    def __enter__(self):
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._ctx = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            locale="en-US",
            timezone_id="Europe/London",
        )
        self._ctx.route(
            "**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,mp4,webp}",
            lambda r: r.abort(),
        )
        self._page = self._ctx.new_page()
        log.info("Browser started")
        return self

    def __exit__(self, *_):
        if self._browser: self._browser.close()
        if self._pw: self._pw.stop()
        log.info("Browser stopped")

    def _sleep(self):
        time.sleep(random.uniform(*self.delay))

    # Navigation
    
    def _load_page1(self, url: str, retries: int = 3) -> bool:
        for attempt in range(retries):
            try:
                self._page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                self._page.wait_for_function(
                    "() => document.querySelectorAll('[data-testid=\"game-row\"]').length > 0",
                    timeout=15_000,
                )
                return True
            except PWTimeout:
                log.warning(f"Timeout loading {url}, attempt {attempt+1}/{retries}")
                self._sleep()
        return False

    def _get_page_count(self) -> int:
        
        try:
            nums = self._page.evaluate("""
                () => Array.from(document.querySelectorAll('.pagination-link'))
                    .map(el => parseInt(el.getAttribute('data-number')))
                    .filter(n => !isNaN(n))
            """)
            return max(nums) if nums else 1
        except Exception:
            return 1

    def _goto_page(self, page_num: int) -> bool:
        
        if page_num == 1:
            return True  # already on page 1

        try:
            
            self._page.click(f'.pagination-link[data-number="{page_num}"]', timeout=5_000)

            
            self._page.wait_for_function(
                "() => document.querySelectorAll('[data-testid=\"game-row\"]').length === 0",
                timeout=8_000,
            )
            
            self._page.wait_for_function(
                "() => document.querySelectorAll('[data-testid=\"game-row\"]').length > 0",
                timeout=15_000,
            )
            return True
        except PWTimeout:
            log.warning(f"Timeout navigating to page {page_num}")
            return False
        except Exception as e:
            log.warning(f"Error navigating to page {page_num}: {e}")
            return False

    # Parsing
    
    def _parse_page(self, competition: str, carry_date: str = '', carry_label: str = '') -> tuple[list[dict], str, str]:
        
        is_neutral_default = competition in ALWAYS_NEUTRAL

        results = self._page.evaluate("""
            (carry) => {
                var DATE_RE = /(\\d{1,2}\\s+[A-Za-z]{3}\\s+\\d{4})/;
                var results = [];
                var currentDate = carry.date;
                var currentRoundLabel = carry.label;

                var firstGameRow = document.querySelector('[data-testid="game-row"]');
                if (!firstGameRow) return [];

                var container = firstGameRow.parentElement
                    && firstGameRow.parentElement.parentElement
                    && firstGameRow.parentElement.parentElement.parentElement;
                if (!container) return [];

                var children = Array.from(container.children);
                // Skip first (breadcrumb) and last (secondary-header)
                var eventRows = children.filter(function(el) {
                    var testid = el.getAttribute('data-testid') || '';
                    return testid !== 'sport-country-league-item' && testid !== 'secondary-header';
                });

                eventRows.forEach(function(eventRow) {
                    var lines = (eventRow.innerText || '').split('\\n').map(function(l) { return l.trim(); });
                    var firstLine = lines[0];
                    var dateMatch = null;
                    var dateLine = '';
                    for (var li = 0; li < Math.min(lines.length, 10); li++) {
                        if (lines[li].match(DATE_RE)) {
                            dateMatch = lines[li].match(DATE_RE);
                            dateLine = lines[li];
                            break;
                        }
                    }
                    if (dateMatch) {
                        currentDate = dateMatch[1];
                        var dashIdx = dateLine.indexOf(' - ');
                        currentRoundLabel = dashIdx >= 0 ? dateLine.slice(dashIdx + 3).trim() : '';
                    }

                    // Find game-rows inside this eventRow
                    var gameRows = eventRow.querySelectorAll('[data-testid="game-row"]');

                    Array.from(gameRows).forEach(function(gameRow) {
                        // Get direct odd-container children only
                        var directOdds = Array.from(gameRow.children).filter(function(el) {
                            var tid = el.getAttribute('data-testid') || '';
                            return tid === 'odd-container-default' || tid === 'odd-container-winning';
                        });

                        // Skip second game-row (duplicate, 0 odds)
                        if (directOdds.length === 0) return;

                        // Teams
                        var teamEls = gameRow.querySelectorAll('.participant-name');
                        if (teamEls.length < 2) return;
                        var homeTeam = teamEls[0].innerText.trim();
                        var awayTeam = teamEls[1].innerText.trim();

                        // Separate odds by testid
                        if (directOdds.length !== 3) return;

                        function parseOdd(el) {
                            if (!el) return null;
                            var v = parseFloat(el.innerText.trim());
                            return (!isNaN(v) && v >= 1.01 && v <= 300) ? v : null;
                        }

                        results.push({
                            date_raw:    currentDate,
                            round_label: currentRoundLabel,
                            home_team:   homeTeam,
                            away_team:   awayTeam,
                            home_odd:    parseOdd(directOdds[0]),
                            draw_odd:    parseOdd(directOdds[1]),
                            away_odd:    parseOdd(directOdds[2]),
                        });
                    });
                });

                return {rows: results, lastDate: currentDate, lastLabel: currentRoundLabel};
            }
        """,
            {"date": carry_date, "label": carry_label}
        )
        last_date = results.get("lastDate", carry_date)
        last_label = results.get("lastLabel", carry_label)
        results = results.get("rows", [])

        cleaned = []
        for r in results:
            date_str = _parse_date(r.get("date_raw", ""))
            if not date_str:
                continue

            home = r.get("home_team", "").strip()
            away = r.get("away_team", "").strip()
            if not home or not away:
                continue

            
            round_label = r.get("round_label", "")
            if QUAL_RE.search(round_label):
                actual_comp = f"{competition} qualification"
                neutral = False
            else:
                actual_comp = competition
                neutral = is_neutral_default

        
            if home not in TARGET_TEAMS and away not in TARGET_TEAMS:
                continue

            cleaned.append({
                "date":       date_str,
                "tournament": actual_comp,
                "home_team":  home,
                "away_team":  away,
                "neutral":    neutral,
                "home_odd":   r.get("home_odd"),
                "draw_odd":   r.get("draw_odd"),
                "away_odd":   r.get("away_odd"),
            })
        return cleaned, last_date, last_label

    # Public
    
    def scrape_edition(
        self,
        region: str,
        slug: str,
        competition: str,
        from_year: int = 2000,
    ) -> list[dict]:
        url = f"{BASE_URL}/football/{region}/{slug}/results/"
        log.info(f"Scraping: {competition} | {url}")

        if not self._load_page1(url):
            log.warning(f"Failed to load: {url}")
            return []

        title = self._page.title().lower()
        if "not found" in title or "404" in title:
            log.warning(f"404: {url}")
            return []

        n_pages = self._get_page_count()
        log.info(f"  {n_pages} page(s)")

        all_rows: list[dict] = []
        carry_date, carry_label = "", ""
        
        
        for pg in range(1, n_pages + 1):
            if pg > 1:
                if not self._goto_page(pg):
                    log.warning(f"  Failed to navigate to page {pg}, skipping")
                    continue
                self._sleep()

            rows, carry_date, carry_label = self._parse_page(competition, carry_date, carry_label)  

            if from_year > 2000:
                rows = [r for r in rows if r["date"][:4] >= str(from_year)]

            log.info(f"  Page {pg}/{n_pages}: {len(rows)} matches")
            all_rows.extend(rows)

            if pg < n_pages:
                self._sleep()

        return all_rows


# Utilities

def _parse_date(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    m = re.search(r"(\d{4}-\d{2}-\d{2})", raw)
    if m:
        return m.group(1)
    m = re.search(r"(\d{1,2})\s+([A-Za-z]{3})\s+(\d{2,4})", raw)
    if m:
        try:
            d, mon, y = m.group(1), m.group(2), m.group(3)
            if len(y) == 2:
                y = "20" + y
            return datetime.strptime(f"{d} {mon} {y}", "%d %b %Y").strftime("%Y-%m-%d")
        except ValueError:
            pass
    return ""


def _save(rows: list[dict], path: str):
    if not rows:
        return
    df = pd.DataFrame(rows, columns=[
        "date", "tournament", "home_team", "away_team",
        "neutral", "home_odd", "draw_odd", "away_odd",
    ])
    df = df.drop_duplicates(subset=["date", "home_team", "away_team", "tournament"])
    df.sort_values(["tournament", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(path, index=False)
    log.info(f"Saved {len(df):,} rows -> {path}")


# Orchestrator

def run_scraper(
    competitions: list[str] | None = None,
    from_year: int = 2000,
    output_path: str = None,
    resume: bool = False,
    headless: bool = True,
    delay_range: tuple = (2.0, 4.0),
):
    selected = {k: v for k, v in COMPETITIONS.items()
                if not competitions or k in competitions}
    
    if output_path is None:
        ROOT = Path(__file__).resolve().parents[2]
        output_path = ROOT / "data" / "raw" / "odds_raw.csv"
        
    if competitions:
        unknown = set(competitions) - set(COMPETITIONS)
        if unknown:
            log.warning(f"Unknown competitions (skipped): {unknown}")

    done: set[tuple[str, str]] = set()
    all_rows: list[dict] = []

    if resume and Path(output_path).exists():
        existing = pd.read_csv(output_path)
        all_rows = existing.to_dict("records")
        for _, row in existing.iterrows():
            done.add((row["tournament"], str(row["date"])[:4]))
        log.info(f"Resuming from {len(all_rows):,} existing rows")

    total = sum(len(v) for v in selected.values())
    log.info(f"Starting: {len(selected)} competitions, {total} editions")

    with OddsPortalScraper(headless=headless, delay=delay_range) as scraper:
        pbar = tqdm(total=total, unit="ed")

        for comp, editions in selected.items():
            for region, slug in editions:
                pbar.set_description(f"{comp[:35]} / {slug}")

                year_m = re.search(r"(\d{4})", slug)
                year_str = year_m.group(1) if year_m else "????"

                if year_m and int(year_str) < from_year:
                    pbar.update(1)
                    continue

                if resume and (comp, year_str) in done:
                    log.info(f"Skip (done): {comp} {year_str}")
                    pbar.update(1)
                    continue

                try:
                    rows = scraper.scrape_edition(region, slug, comp, from_year)
                    all_rows.extend(rows)
                    log.info(f"  -> {len(rows)} matches collected")
                    _save(all_rows, output_path)
                except Exception as e:
                    log.error(f"Error {comp}/{slug}: {e}", exc_info=True)

                pbar.update(1)
                scraper._sleep()

        pbar.close()

    _save(all_rows, output_path)
    log.info(f"Done. {len(all_rows):,} total rows -> {output_path}")


# CLI

def main():
    ap = argparse.ArgumentParser(
        description="Scrape OddsPortal international football odds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scraper.py
  python scraper.py --competitions "FIFA World Cup" "UEFA Euro" "Copa America"
  python scraper.py --from-year 2006 --output wc_euro.csv
  python scraper.py --resume --headful
  python scraper.py --list-competitions
""",
    )
    ap.add_argument("--competitions", nargs="+", metavar="NAME")
    ap.add_argument("--from-year", type=int, default=2000, metavar="YEAR")
    ap.add_argument("--output", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--delay", type=float, nargs=2, default=[2.0, 4.0],
                    metavar=("MIN", "MAX"))
    ap.add_argument("--list-competitions", action="store_true")
    args = ap.parse_args()

    if args.list_competitions:
        print("\nAvailable competitions:\n")
        for name, editions in COMPETITIONS.items():
            print(f"  {name!r}  ({len(editions)} editions)")
        print()
        return

    run_scraper(
        competitions=args.competitions,
        from_year=args.from_year,
        output_path=args.output,
        resume=args.resume,
        headless=not args.headful,
        delay_range=tuple(args.delay),
    )


if __name__ == "__main__":
    main()