#!/bin/bash
# Helper script to create a new release

if [ -z "$1" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 1.0.0"
    echo ""
    echo "This will:"
    echo "  1. Create and push a git tag v<version>"
    echo "  2. Trigger GitHub Actions to build and release"
    exit 1
fi

VERSION="$1"
TAG="v$VERSION"

echo "🏷️  Creating release $TAG..."

# Check if tag already exists
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "❌ Tag $TAG already exists!"
    exit 1
fi

# Check if we're on main or release branch
BRANCH=$(git branch --show-current)
if [[ "$BRANCH" != "main" && "$BRANCH" != "release" ]]; then
    echo "⚠️  Warning: You're on branch '$BRANCH'. Consider switching to 'main' or 'release'."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create and push the tag
echo "📝 Creating tag $TAG..."
git tag -a "$TAG" -m "Release $TAG"

echo "🚀 Pushing tag to GitHub..."
git push origin "$TAG"

echo "✅ Done! GitHub Actions will now:"
echo "   - Build executables for Windows, macOS, and Linux"
echo "   - Create a GitHub release with downloadable files"
echo "   - You can monitor progress at: https://github.com/$(git config remote.origin.url | sed 's/.*github.com[:/]\([^.]*\).*/\1/')/actions"

echo ""
echo "🔗 Release will be available at:"
echo "   https://github.com/$(git config remote.origin.url | sed 's/.*github.com[:/]\([^.]*\).*/\1/')/releases/tag/$TAG"