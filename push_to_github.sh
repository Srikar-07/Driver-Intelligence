#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  Drivr Intelligence — GitHub Push Script
#  Run this once after creating your GitHub repo
# ─────────────────────────────────────────────────────────────

set -e

echo ""
echo "🚗 Drivr Intelligence — GitHub Push"
echo "────────────────────────────────────"
echo ""

# ── Step 1: Get GitHub username ───────────────────────────────
read -p "Enter your GitHub username: " GITHUB_USER

# ── Step 2: Confirm repo name ─────────────────────────────────
REPO_NAME="drivr-intelligence"
echo ""
echo "This will push to: https://github.com/$GITHUB_USER/$REPO_NAME"
read -p "Press Enter to continue (or Ctrl+C to cancel)..."

# ── Step 3: Add remote & push ─────────────────────────────────
echo ""
echo "Adding remote origin..."
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"

echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Done! Your repo is live at:"
echo "   https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
echo "Next steps:"
echo "  1. Add a description: 'End-to-end ML system — pricing, demand forecasting, matching, fraud detection'"
echo "  2. Add topics: machine-learning, python, xgboost, fastapi, streamlit, recommender-system, fraud-detection"
echo "  3. Pin it to your GitHub profile"
echo ""
