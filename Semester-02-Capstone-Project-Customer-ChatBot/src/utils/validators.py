"""Validation utilities for the application."""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


def validate_github_token(token: str) -> bool:
    """Validate GitHub personal access token format.

    Args:
        token: GitHub token to validate

    Returns:
        True if token format is valid
    """
    if not token:
        return False

    # GitHub tokens start with 'ghp_' for personal access tokens
    # or 'github_pat_' for fine-grained tokens
    pattern = r'^(ghp_[a-zA-Z0-9]{36}|github_pat_[a-zA-Z0-9_]{82})$'
    return bool(re.match(pattern, token))


def validate_github_repo(repo: str) -> bool:
    """Validate GitHub repository format.

    Args:
        repo: Repository string in format 'owner/repo'

    Returns:
        True if repository format is valid
    """
    if not repo or '/' not in repo:
        return False

    parts = repo.split('/')
    if len(parts) != 2:
        return False

    owner, repo_name = parts

    # GitHub username/org and repo name validation
    # Allow alphanumeric, hyphens, underscores
    pattern = r'^[a-zA-Z0-9._-]+$'

    return (
        bool(re.match(pattern, owner)) and
        bool(re.match(pattern, repo_name)) and
        len(owner) > 0 and len(repo_name) > 0 and
        not owner.startswith('-') and not owner.endswith('-') and
        not repo_name.startswith('-') and not repo_name.endswith('-')
    )


def validate_email(email: str) -> bool:
    """Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        True if email format is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """Validate URL format.

    Args:
        url: URL to validate

    Returns:
        True if URL format is valid
    """
    pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/.*)?$'
    return bool(re.match(pattern, url))


def validate_phone(phone: str) -> bool:
    """Validate phone number format.

    Args:
        phone: Phone number to validate

    Returns:
        True if phone format is valid
    """
    # Support various international formats
    patterns = [
        r'^\+\d{1,3}-\d{3,4}-\d{3,4}$',  # +1-555-0123
        r'^\+\d{1,3}\s\d{3,4}\s\d{3,4}$',  # +1 555 0123
        r'^\+\d{1,3}\(\d{3,4}\)\d{3,4}$',  # +1(555)0123
        r'^\+\d{10,15}$',  # +15550123456
        r'^\d{3}-\d{3}-\d{4}$',  # 555-123-4567
        r'^\(\d{3}\)\s?\d{3}-\d{4}$',  # (555) 123-4567
    ]

    return any(re.match(pattern, phone) for pattern in patterns)


def validate_document_file(file_path: Path) -> Tuple[bool, Optional[str]]:
    """Validate document file for processing.

    Args:
        file_path: Path to document file

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"

    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"

    # Check file size (max 100MB)
    max_size = 100 * 1024 * 1024  # 100MB
    if file_path.stat().st_size > max_size:
        return False, f"File too large: {file_path.stat().st_size / 1024 / 1024:.1f}MB (max 100MB)"

    # Check file extension
    supported_extensions = {'.pdf', '.txt', '.md', '.docx', '.doc'}
    if file_path.suffix.lower() not in supported_extensions:
        return False, f"Unsupported file type: {file_path.suffix} (supported: {', '.join(supported_extensions)})"

    return True, None


def validate_documents_directory(data_path: Path) -> Tuple[bool, List[str]]:
    """Validate documents directory meets requirements.

    Args:
        data_path: Path to documents directory

    Returns:
        Tuple of (meets_requirements, list_of_issues)
    """
    issues = []

    if not data_path.exists():
        issues.append(f"Documents directory does not exist: {data_path}")
        return False, issues

    if not data_path.is_dir():
        issues.append(f"Documents path is not a directory: {data_path}")
        return False, issues

    # Get all document files
    document_files = []
    for extension in ['.pdf', '.txt', '.md', '.docx', '.doc']:
        document_files.extend(data_path.glob(f'**/*{extension}'))

    if len(document_files) < 3:
        issues.append(f"Need at least 3 documents, found {len(document_files)}")

    # Count PDFs
    pdf_files = [f for f in document_files if f.suffix.lower() == '.pdf']
    if len(pdf_files) < 2:
        issues.append(f"Need at least 2 PDF documents, found {len(pdf_files)}")

    # Check for 400+ page PDF (approximate check by file size)
    large_pdf_found = False
    for pdf_file in pdf_files:
        # Rough estimate: 400 pages ≈ 10MB+ for text-heavy PDFs
        if pdf_file.stat().st_size > 10 * 1024 * 1024:  # 10MB
            large_pdf_found = True
            break

    if not large_pdf_found:
        issues.append("Need at least 1 PDF with 400+ pages (or 10MB+ file size)")

    # Validate each file
    for doc_file in document_files:
        is_valid, error = validate_document_file(doc_file)
        if not is_valid:
            issues.append(f"Invalid document {doc_file.name}: {error}")

    return len(issues) == 0, issues


def validate_query(query: str) -> Tuple[bool, Optional[str]]:
    """Validate user query.

    Args:
        query: User query string

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"

    if len(query.strip()) < 3:
        return False, "Query too short (minimum 3 characters)"

    if len(query) > 1000:
        return False, "Query too long (maximum 1000 characters)"

    # Check for potentially harmful content
    harmful_patterns = [
        r'<script[^>]*>',  # Script tags
        r'javascript:',     # JavaScript URLs
        r'data:text/html',  # Data URLs
    ]

    for pattern in harmful_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False, "Query contains potentially harmful content"

    return True, None


def validate_configuration() -> Tuple[bool, List[str]]:
    """Validate application configuration.

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    from config import settings

    issues = []

    # Validate GitHub settings
    if not validate_github_token(settings.github_token):
        issues.append("Invalid GitHub token format")

    if not validate_github_repo(settings.github_repo):
        issues.append("Invalid GitHub repository format (should be 'owner/repo')")

    # Validate company information
    if not validate_email(settings.company_email):
        issues.append("Invalid company email format")

    if not validate_url(settings.company_website):
        issues.append("Invalid company website URL format")

    if not validate_phone(settings.company_phone):
        issues.append("Invalid company phone format")

    # Validate paths
    if not settings.data_raw_path.exists():
        issues.append(f"Raw data directory does not exist: {settings.data_raw_path}")

    # Validate numeric settings
    if settings.max_chunk_size <= 0:
        issues.append("max_chunk_size must be positive")

    if settings.chunk_overlap < 0:
        issues.append("chunk_overlap cannot be negative")

    if settings.chunk_overlap >= settings.max_chunk_size:
        issues.append("chunk_overlap must be less than max_chunk_size")

    if settings.top_k_retrieval <= 0:
        issues.append("top_k_retrieval must be positive")

    if not 0 <= settings.similarity_threshold <= 1:
        issues.append("similarity_threshold must be between 0 and 1")

    logger.info(
        "Configuration validation completed",
        valid=len(issues) == 0,
        issues_count=len(issues)
    )

    return len(issues) == 0, issues