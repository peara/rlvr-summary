#!/bin/bash

# GitHub Issues Creation Script
# This script automatically creates GitHub issues from the quick_issues.md file
# using the GitHub CLI (gh)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}Error: GitHub CLI (gh) is not installed.${NC}"
    echo "Please install it from https://cli.github.com/"
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}Error: Not in a git repository.${NC}"
    exit 1
fi

# Check if user is authenticated with gh
if ! gh auth status &> /dev/null; then
    echo -e "${RED}Error: Not authenticated with GitHub CLI.${NC}"
    echo "Please run: gh auth login"
    exit 1
fi

# Check if quick_issues.md exists
if [[ ! -f "quick_issues.md" ]]; then
    echo -e "${RED}Error: quick_issues.md file not found.${NC}"
    exit 1
fi

echo -e "${BLUE}🚀 RLVR Summary - GitHub Issues Creation Script${NC}"
echo -e "${BLUE}===============================================${NC}"
echo

# Parse and create issues
create_issues() {
    local issue_count=0
    local title=""
    local labels=""
    local body=""
    local in_issue=false

    echo -e "${YELLOW}📝 Parsing issues from quick_issues.md...${NC}"
    
    # Read the file and process each issue
    while IFS= read -r line; do
        # Check if this is an issue title
        if [[ "$line" =~ ^###[[:space:]]*Issue:[[:space:]]*(.*) ]]; then
            # If we were already processing an issue, create it first
            if [[ "$in_issue" == true && -n "$title" ]]; then
                create_single_issue "$issue_count" "$title" "$labels" "$body"
            fi
            
            # Start new issue
            issue_count=$((issue_count + 1))
            title="${BASH_REMATCH[1]}"
            labels=""
            body=""
            in_issue=true
            
        # Check if this is the labels line
        elif [[ "$line" =~ ^\*\*Labels:\*\*[[:space:]]*(.*) ]]; then
            labels="${BASH_REMATCH[1]}"
            
        # Check if this is a separator (end of issue)
        elif [[ "$line" =~ ^---$ ]]; then
            # Create the issue if we have one
            if [[ "$in_issue" == true && -n "$title" ]]; then
                create_single_issue "$issue_count" "$title" "$labels" "$body"
            fi
            
            # Reset for next issue
            title=""
            labels=""
            body=""
            in_issue=false
            
        # Otherwise, add to body (but skip the labels line and title line)
        elif [[ "$in_issue" == true && ! "$line" =~ ^###[[:space:]]*Issue: && ! "$line" =~ ^\*\*Labels:\*\* ]]; then
            # Add to body (preserve empty lines within the issue)
            body+="$line"$'\n'
        fi
    done < quick_issues.md
    
    # Handle the last issue if there's no final separator
    if [[ "$in_issue" == true && -n "$title" ]]; then
        create_single_issue "$issue_count" "$title" "$labels" "$body"
    fi
    
    echo -e "${GREEN}🎉 Completed! Created $issue_count issues.${NC}"
}

# Function to create a single issue
create_single_issue() {
    local issue_num="$1"
    local title="$2"
    local labels="$3"
    local body="$4"
    
    echo -e "${BLUE}Creating issue $issue_num: ${title}${NC}"
    
    # Convert labels format for gh cli (comma-separated to space-separated with --label flags)
    local label_args=""
    if [[ -n "$labels" ]]; then
        IFS=',' read -ra LABEL_ARRAY <<< "$labels"
        for label in "${LABEL_ARRAY[@]}"; do
            # Trim whitespace
            label=$(echo "$label" | xargs)
            label_args+="--label \"$label\" "
        done
    fi
    
    # Clean up body (remove leading/trailing whitespace)
    body=$(echo "$body" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    
    # Create the issue using gh cli
    local gh_command="gh issue create --title \"$title\" --body \"$body\" $label_args"
    
    echo -e "${GREEN}  Labels: $labels${NC}"
    
    # Execute the command
    if eval "$gh_command"; then
        echo -e "${GREEN}  ✅ Issue created successfully${NC}"
    else
        echo -e "${RED}  ❌ Failed to create issue${NC}"
    fi
    echo
}

# Function to create recommended labels first
create_labels() {
    echo -e "${YELLOW}🏷️ Creating recommended labels...${NC}"
    
    # Define labels with colors
    declare -A labels
    labels[setup]="0052cc"
    labels[infrastructure]="1d76db"
    labels[phase-0]="fbca04"
    labels[phase-a]="ff6b6b"
    labels[phase-b]="4ecdc4"
    labels[phase-c]="45b7d1"
    labels[phase-d]="f9ca24"
    labels[phase-e]="6c5ce7"
    labels[high-priority]="d73a49"
    labels[medium-priority]="f9ca24"
    labels[low-priority]="28a745"
    labels[data-pipeline]="0e8a16"
    labels[rewards]="b60205"
    labels[rl]="ff9f40"
    labels[training]="ff6347"
    labels[evaluation]="8a2be2"
    labels[monitoring]="008080"
    labels[fenice]="ff1493"
    labels[synthetic-data]="32cd32"
    labels[tools]="8b4513"
    labels[distillation]="dda0dd"
    labels[deployment]="2f4f4f"
    labels[documentation]="5dade2"
    
    for label in "${!labels[@]}"; do
        if ! gh label list --limit 100 | grep -q "^$label"; then
            echo -e "${GREEN}  Creating label: $label${NC}"
            gh label create "$label" --color "${labels[$label]}" --description "Auto-created for RLVR Summary project" 2>/dev/null || true
        else
            echo -e "${YELLOW}  Label already exists: $label${NC}"
        fi
    done
    echo
}

# Main execution
main() {
    echo -e "${YELLOW}Do you want to create the recommended labels first? (y/n):${NC}"
    read -r create_labels_answer
    
    if [[ "$create_labels_answer" =~ ^[Yy]$ ]]; then
        create_labels
    fi
    
    echo -e "${YELLOW}Ready to create issues from quick_issues.md. Continue? (y/n):${NC}"
    read -r answer
    
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        create_issues
    else
        echo -e "${YELLOW}Aborted by user.${NC}"
        exit 0
    fi
    
    echo -e "${GREEN}🎯 Next steps:${NC}"
    echo -e "${GREEN}  1. Check the created issues on GitHub${NC}"
    echo -e "${GREEN}  2. Assign team members to issues${NC}"
    echo -e "${GREEN}  3. Follow the dependency order from issue_creation_order.md${NC}"
    echo -e "${GREEN}  4. Start with Setup Phase issues first${NC}"
}

main "$@"