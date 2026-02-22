#!/usr/bin/env python3
"""
Traffic Speed Analysis Report Generator (Multi-Day)
Reads car_log.csv and produces a PDF report with overall and per-day statistics and charts.

Usage:
    python generate_report.py                        # uses car_log.csv in current dir
    python generate_report.py --input my_log.csv     # specify input file
    python generate_report.py --output report.pdf    # specify output file
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# --- Colors ---
COLOR_PRIMARY = "#2c3e50"
COLOR_RIGHT = "#e74c3c"
COLOR_LEFT = "#3498db"
COLOR_GRID = "#ecf0f1"
COLOR_ACCENT = "#2ecc71"


def load_data(csv_path):
    """Load and parse the CSV log file."""
    records = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rec = {
                    "timestamp": datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S"),
                    "car_id": int(row["car_id"]),
                    "direction": row["direction"].strip(),
                    "speed_mph": float(row["speed_mph"]),
                    "speed_px_s": float(row["speed_px_s"]),
                    "num_points": int(row["num_points"]),
                    "duration_sec": float(row["duration_sec"]),
                    "entry_x": int(row["entry_x"]),
                    "exit_x": int(row["exit_x"]),
                }
                records.append(rec)
            except (ValueError, KeyError):
                continue
    return records


def compute_stats(speeds):
    """Compute summary statistics for a list of speeds."""
    if not speeds:
        return None
    arr = np.array(speeds)
    return {
        "count": len(arr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p85": float(np.percentile(arr, 85)),
        "p95": float(np.percentile(arr, 95)),
    }


def group_by_day(records):
    """Group records by date string, returned as ordered dict."""
    days = defaultdict(list)
    for r in records:
        day_key = r["timestamp"].strftime("%Y-%m-%d")
        days[day_key].append(r)
    return dict(sorted(days.items()))


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

def make_histogram(all_speeds, right_speeds, left_speeds, filepath, title_suffix=""):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.arange(0, max(all_speeds + [1]) + 5, 2)
    if right_speeds:
        ax.hist(right_speeds, bins=bins, alpha=0.65, label="Right",
                color=COLOR_RIGHT, edgecolor="white", linewidth=0.5)
    if left_speeds:
        ax.hist(left_speeds, bins=bins, alpha=0.65, label="Left",
                color=COLOR_LEFT, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Speed (mph)", fontsize=10)
    ax.set_ylabel("Number of Cars", fontsize=10)
    title = "Speed Distribution by Direction"
    if title_suffix:
        title += f" \u2014 {title_suffix}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def make_hourly_chart(records, filepath, title_suffix=""):
    hourly_right = defaultdict(int)
    hourly_left = defaultdict(int)
    for r in records:
        hour = r["timestamp"].replace(minute=0, second=0, microsecond=0)
        if r["direction"] == "RIGHT":
            hourly_right[hour] += 1
        else:
            hourly_left[hour] += 1
    all_hours = sorted(set(list(hourly_right.keys()) + list(hourly_left.keys())))
    if not all_hours:
        return None
    right_counts = [hourly_right.get(h, 0) for h in all_hours]
    left_counts = [hourly_left.get(h, 0) for h in all_hours]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bar_width = timedelta(minutes=20)
    ax.bar(all_hours, right_counts, width=bar_width, alpha=0.75,
           label="Right", color=COLOR_RIGHT, edgecolor="white", linewidth=0.5)
    ax.bar([h + bar_width for h in all_hours], left_counts, width=bar_width,
           alpha=0.75, label="Left", color=COLOR_LEFT, edgecolor="white", linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Hour", fontsize=10)
    ax.set_ylabel("Car Count", fontsize=10)
    title = "Traffic Volume by Hour"
    if title_suffix:
        title += f" \u2014 {title_suffix}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def make_speed_over_time(records, filepath, title_suffix=""):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    right = [(r["timestamp"], r["speed_mph"]) for r in records if r["direction"] == "RIGHT"]
    left = [(r["timestamp"], r["speed_mph"]) for r in records if r["direction"] == "LEFT"]
    if right:
        ax.scatter([r[0] for r in right], [r[1] for r in right],
                   alpha=0.5, s=20, color=COLOR_RIGHT, label="Right", edgecolors="none")
    if left:
        ax.scatter([r[0] for r in left], [r[1] for r in left],
                   alpha=0.5, s=20, color=COLOR_LEFT, label="Left", edgecolors="none")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Speed (mph)", fontsize=10)
    title = "Speed Over Time"
    if title_suffix:
        title += f" \u2014 {title_suffix}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def make_cumulative_chart(all_speeds, filepath, title_suffix=""):
    fig, ax = plt.subplots(figsize=(7, 3))
    sorted_speeds = np.sort(all_speeds)
    cumulative = np.arange(1, len(sorted_speeds) + 1) / len(sorted_speeds) * 100
    ax.plot(sorted_speeds, cumulative, color=COLOR_PRIMARY, linewidth=2)
    ax.fill_between(sorted_speeds, cumulative, alpha=0.1, color=COLOR_PRIMARY)
    ax.set_xlabel("Speed (mph)", fontsize=10)
    ax.set_ylabel("% of Cars At or Below", fontsize=10)
    title = "Cumulative Speed Distribution"
    if title_suffix:
        title += f" \u2014 {title_suffix}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def make_daily_comparison_chart(daily_groups, filepath):
    """Bar chart comparing total cars and median speed across days."""
    days = list(daily_groups.keys())
    totals = [len(recs) for recs in daily_groups.values()]
    medians = [float(np.median([r["speed_mph"] for r in recs])) for recs in daily_groups.values()]

    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(days))
    width = 0.38

    ax1.bar(x - width / 2, totals, width, label="Total Cars",
            color=COLOR_RIGHT, alpha=0.8, edgecolor="white")
    ax1.set_ylabel("Total Cars", fontsize=10, color=COLOR_RIGHT)
    ax1.tick_params(axis="y", labelcolor=COLOR_RIGHT)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, medians, width, label="Median mph",
            color=COLOR_LEFT, alpha=0.8, edgecolor="white")
    ax2.set_ylabel("Median Speed (mph)", fontsize=10, color=COLOR_LEFT)
    ax2.tick_params(axis="y", labelcolor=COLOR_LEFT)

    ax1.set_xticks(x)
    day_labels = []
    for d in days:
        dt = datetime.strptime(d, "%Y-%m-%d")
        day_labels.append(dt.strftime("%a\n%m/%d"))
    ax1.set_xticklabels(day_labels, fontsize=9)
    ax1.set_title("Daily Comparison: Volume & Median Speed", fontsize=12, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    ax1.grid(axis="y", alpha=0.2)
    ax1.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def format_stat_row(label, stats):
    if not stats:
        return [label, "\u2014", "\u2014", "\u2014", "\u2014", "\u2014", "\u2014", "\u2014"]
    return [
        label,
        str(stats["count"]),
        f"{stats['median']:.1f}",
        f"{stats['mean']:.1f}",
        f"{stats['p25']:.1f}",
        f"{stats['p75']:.1f}",
        f"{stats['p85']:.1f}",
        f"{stats['max']:.1f}",
    ]


def build_speed_table(all_stats, right_stats, left_stats):
    table_data = [
        ["", "Cars", "Median", "Mean", "25th %", "75th %", "85th %", "Max"],
        format_stat_row("All", all_stats),
        format_stat_row("Right", right_stats),
        format_stat_row("Left", left_stats),
    ]
    t = Table(table_data, colWidths=[70, 45, 55, 55, 55, 55, 55, 55])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(COLOR_PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#bdc3c7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor(COLOR_GRID)]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def build_hourly_table(records):
    hourly_data = defaultdict(list)
    for r in records:
        hour_key = r["timestamp"].strftime("%H:00")
        hourly_data[hour_key].append(r)
    if not hourly_data:
        return None
    hourly_table_data = [["Hour", "Total", "Right", "Left", "Median mph", "Max mph"]]
    for hour in sorted(hourly_data.keys()):
        hrs = hourly_data[hour]
        hr_speeds = [r["speed_mph"] for r in hrs]
        hr_right = sum(1 for r in hrs if r["direction"] == "RIGHT")
        hr_left = sum(1 for r in hrs if r["direction"] == "LEFT")
        hourly_table_data.append([
            hour, str(len(hrs)), str(hr_right), str(hr_left),
            f"{np.median(hr_speeds):.1f}", f"{np.max(hr_speeds):.1f}"
        ])
    ht = Table(hourly_table_data, colWidths=[60, 50, 50, 50, 70, 70])
    ht.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(COLOR_PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#bdc3c7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor(COLOR_GRID)]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return ht


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def add_stats_section(story, records, styles, chart_dir, prefix):
    """Add speed stats table, volume text, and data quality for a set of records."""
    all_speeds = [r["speed_mph"] for r in records]
    right_speeds = [r["speed_mph"] for r in records if r["direction"] == "RIGHT"]
    left_speeds = [r["speed_mph"] for r in records if r["direction"] == "LEFT"]

    all_stats = compute_stats(all_speeds)
    right_stats = compute_stats(right_speeds)
    left_stats = compute_stats(left_speeds)

    story.append(Paragraph("Speed Summary", styles["SectionHead"]))
    story.append(build_speed_table(all_stats, right_stats, left_stats))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "All speeds in mph. The 85th percentile is commonly used for traffic engineering decisions.",
        styles["BodyText2"]
    ))

    timestamps = [r["timestamp"] for r in records]
    duration_hrs = (max(timestamps) - min(timestamps)).total_seconds() / 3600
    duration_hrs = max(duration_hrs, 0.001)

    total = len(records)
    right_count = len(right_speeds)
    left_count = len(left_speeds)
    cars_per_hr = total / duration_hrs

    story.append(Paragraph("Traffic Volume", styles["SectionHead"]))
    story.append(Paragraph(
        f"Total vehicles: {total} ({right_count} rightbound, {left_count} leftbound). "
        f"Average throughput: {cars_per_hr:.1f} cars/hour overall, "
        f"{right_count / duration_hrs:.1f}/hr rightbound, "
        f"{left_count / duration_hrs:.1f}/hr leftbound.",
        styles["BodyText2"]
    ))

    avg_points = np.mean([r["num_points"] for r in records])
    avg_duration = np.mean([r["duration_sec"] for r in records])
    story.append(Paragraph("Data Quality", styles["SectionHead"]))
    story.append(Paragraph(
        f"Average tracking points per car: {avg_points:.1f}. "
        f"Average tracking duration: {avg_duration:.2f}s. "
        f"More points and longer durations yield more reliable speed estimates.",
        styles["BodyText2"]
    ))


def add_charts_section(story, records, styles, chart_dir, prefix, title_suffix=""):
    """Add histogram, cumulative, hourly, speed-over-time, and hourly table."""
    all_speeds = [r["speed_mph"] for r in records]
    right_speeds = [r["speed_mph"] for r in records if r["direction"] == "RIGHT"]
    left_speeds = [r["speed_mph"] for r in records if r["direction"] == "LEFT"]

    story.append(Paragraph("Speed Distribution", styles["SectionHead"]))
    hist_path = os.path.join(chart_dir, f"{prefix}_histogram.png")
    make_histogram(all_speeds, right_speeds, left_speeds, hist_path, title_suffix)
    story.append(Image(hist_path, width=6.5 * inch, height=3.25 * inch))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Cumulative Speed Distribution", styles["SectionHead"]))
    cdf_path = os.path.join(chart_dir, f"{prefix}_cumulative.png")
    make_cumulative_chart(all_speeds, cdf_path, title_suffix)
    story.append(Image(cdf_path, width=6.5 * inch, height=2.8 * inch))

    story.append(PageBreak())

    story.append(Paragraph("Traffic Volume by Hour", styles["SectionHead"]))
    hourly_path = os.path.join(chart_dir, f"{prefix}_hourly.png")
    result = make_hourly_chart(records, hourly_path, title_suffix)
    if result:
        story.append(Image(hourly_path, width=6.5 * inch, height=3.25 * inch))
    else:
        story.append(Paragraph("Not enough data for hourly chart.", styles["BodyText2"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Speed Over Time", styles["SectionHead"]))
    scatter_path = os.path.join(chart_dir, f"{prefix}_scatter.png")
    make_speed_over_time(records, scatter_path, title_suffix)
    story.append(Image(scatter_path, width=6.5 * inch, height=3.25 * inch))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Hourly Breakdown", styles["SectionHead"]))
    ht = build_hourly_table(records)
    if ht:
        story.append(ht)
    else:
        story.append(Paragraph("No hourly data available.", styles["BodyText2"]))


# ---------------------------------------------------------------------------
# Main PDF builder
# ---------------------------------------------------------------------------

def build_pdf(records, output_path, chart_dir):
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"],
        fontSize=22, textColor=HexColor(COLOR_PRIMARY), spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="DayTitle", parent=styles["Heading1"],
        fontSize=18, textColor=HexColor(COLOR_PRIMARY), spaceBefore=0, spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="SectionHead", parent=styles["Heading2"],
        fontSize=14, textColor=HexColor(COLOR_PRIMARY), spaceBefore=16, spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="BodyText2", parent=styles["Normal"],
        fontSize=10, leading=14, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="SubInfo", parent=styles["Normal"],
        fontSize=9, textColor=HexColor("#7f8c8d"), spaceAfter=12, alignment=TA_CENTER
    ))

    story = []
    daily_groups = group_by_day(records)
    num_days = len(daily_groups)

    # ===== TITLE =====
    story.append(Paragraph("Traffic Speed Analysis Report", styles["ReportTitle"]))

    timestamps = [r["timestamp"] for r in records]
    start_time = min(timestamps).strftime("%Y-%m-%d %H:%M")
    end_time = max(timestamps).strftime("%Y-%m-%d %H:%M")
    total_hrs = (max(timestamps) - min(timestamps)).total_seconds() / 3600
    day_word = "day" if num_days == 1 else "days"
    story.append(Paragraph(
        f"Period: {start_time} to {end_time} ({total_hrs:.1f} hours, {num_days} {day_word})"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["SubInfo"]
    ))

    # ===== OVERALL SUMMARY =====
    story.append(Paragraph("Overall Summary", styles["DayTitle"]))
    add_stats_section(story, records, styles, chart_dir, "overall")

    # ===== DAILY COMPARISON (only if multiple days) =====
    if num_days > 1:
        story.append(Paragraph("Daily Comparison", styles["SectionHead"]))

        comp_header = ["Date", "Day", "Cars", "Right", "Left",
                       "Median mph", "85th %", "Max mph"]
        comp_rows = [comp_header]
        for day_str, day_recs in daily_groups.items():
            dt = datetime.strptime(day_str, "%Y-%m-%d")
            speeds = [r["speed_mph"] for r in day_recs]
            st = compute_stats(speeds)
            right_n = sum(1 for r in day_recs if r["direction"] == "RIGHT")
            left_n = sum(1 for r in day_recs if r["direction"] == "LEFT")
            comp_rows.append([
                day_str, dt.strftime("%A"),
                str(st["count"]), str(right_n), str(left_n),
                f"{st['median']:.1f}", f"{st['p85']:.1f}", f"{st['max']:.1f}"
            ])

        ct = Table(comp_rows, colWidths=[72, 65, 42, 42, 42, 62, 52, 55])
        ct.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor(COLOR_PRIMARY)),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (2, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#bdc3c7")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor(COLOR_GRID)]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(ct)
        story.append(Spacer(1, 10))

        comp_chart_path = os.path.join(chart_dir, "daily_comparison.png")
        make_daily_comparison_chart(daily_groups, comp_chart_path)
        story.append(Image(comp_chart_path, width=6.5 * inch, height=3.25 * inch))

    # ===== OVERALL CHARTS =====
    story.append(PageBreak())
    story.append(Paragraph("Overall Charts", styles["DayTitle"]))
    add_charts_section(story, records, styles, chart_dir, "overall", "All Days")

    # ===== PER-DAY SECTIONS =====
    for day_str, day_recs in daily_groups.items():
        dt = datetime.strptime(day_str, "%Y-%m-%d")
        day_label = dt.strftime("%A, %B %d, %Y")

        story.append(PageBreak())
        story.append(Paragraph(day_label, styles["DayTitle"]))
        story.append(Paragraph(
            f"{len(day_recs)} vehicles recorded",
            styles["SubInfo"]
        ))

        add_stats_section(story, day_recs, styles, chart_dir, day_str)
        story.append(PageBreak())
        add_charts_section(story, day_recs, styles, chart_dir, day_str, day_label)

    doc.build(story)
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate traffic speed analysis PDF report")
    parser.add_argument("--input", "-i", default="car_log.csv", help="Path to car_log.csv")
    parser.add_argument("--output", "-o", default="traffic_report.pdf", help="Output PDF path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    records = load_data(args.input)
    if not records:
        print("Error: No valid records found in the log file.")
        sys.exit(1)

    print(f"Loaded {len(records)} records from {args.input}")

    chart_dir = os.path.join(os.path.dirname(args.output) or ".", "_charts")
    os.makedirs(chart_dir, exist_ok=True)

    build_pdf(records, args.output, chart_dir)


if __name__ == "__main__":
    main()
