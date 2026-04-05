# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This directory contains a personal health and fitness data export (June 1, 2025 – March 26, 2026), likely from MyFitnessPal or a similar app. It contains four CSV files — no source code, build system, or tests.

## Data Files

**Exercise-Summary-2025-06-01-to-2026-03-26.csv**
- Columns: Date, Exercise, Type, Exercise Calories, Exercise Minutes, Sets, Reps Per Set, Pounds, Steps, Note
- Primarily cardio (bicycling, aerobics)

**Measurement-Summary-2025-06-01-to-2026-03-26.csv**
- Columns: Date, Weight
- 8 weight entries (Jan 24 – Mar 26, 2026); range ~219 → 209 lbs

**Weight-Datapoints.csv**
- Columns: Date, Weight
- 99 entries (Sep 3, 2025 – Mar 26, 2026); range ~250 → 209 lbs
- More frequent/granular than Measurement-Summary; described as weight estimates

**Nutrition-Summary-2025-06-01-to-2026-03-26.csv**
- Columns: Date, Meal, Calories, Fat, Saturated Fat, Polyunsaturated Fat, Monounsaturated Fat, Trans Fat, Cholesterol, Sodium, Potassium, Carbohydrates, Fiber, Sugar, Protein, Vitamin A, Vitamin C, Calcium, Iron, Note
- Meal-level entries (Breakfast, Lunch, Dinner, Snacks) starting Sep 3, 2025
