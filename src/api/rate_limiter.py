"""
Rate limiting
Limits are applied per API key
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def get_api_key_or_ip(request: Request) -> str:
  """
  Get rate limit key from API key header, fallback to IP
  This ensures rate limiting is per user, not per IP
  """
  # get API key from header
  api_key = request.headers.get("X-API-key")

  if api_key:
    return f"key: {api_key[:16]}"

  return f"ip: {get_remote_address(request)}"

# create limiter instance
limiter = Limiter(key_func=get_api_key_or_ip)

# rate limit constants
RATE_LIMIT_QUERY = "50/minute"  # 50 queries per minute
RATE_LIMIT_UPLOAD = "5/minute"  # 5 uploads per minute
RATE_LIMIT_SESSION = "20/minute"  # 20 sessions per minute
RATE_LIMIT_DEFAULT = "100/minute"  # default limit