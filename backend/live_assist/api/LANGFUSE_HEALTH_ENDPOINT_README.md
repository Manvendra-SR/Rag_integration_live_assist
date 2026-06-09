# Langfuse Health Check Endpoint

## Overview
This document describes the `/health/langfuse` endpoint added to the Live Assist MVP API to check Langfuse observability connectivity and configuration.

## Endpoint Details

**Route:** `GET /health/langfuse`

**Purpose:** Verify Langfuse configuration and connectivity status

**Validates:** Requirements 11.2

## Response Statuses

The endpoint returns one of five possible statuses:

### 1. `healthy`
**Meaning:** Langfuse is properly configured and the connection is successful

**Response Example:**
```json
{
  "status": "healthy",
  "message": "Langfuse connection successful",
  "host": "https://cloud.langfuse.com"
}
```

**Conditions:**
- `LANGFUSE_ENABLED=true`
- `LANGFUSE_PUBLIC_KEY` is set
- `LANGFUSE_SECRET_KEY` is set
- Authentication with Langfuse succeeds

### 2. `disabled`
**Meaning:** Langfuse observability is intentionally disabled

**Response Example:**
```json
{
  "status": "disabled",
  "message": "Langfuse observability is disabled"
}
```

**Conditions:**
- `LANGFUSE_ENABLED=false`

### 3. `misconfigured`
**Meaning:** Credentials are not configured

**Response Example:**
```json
{
  "status": "misconfigured",
  "message": "Langfuse credentials not configured"
}
```

**Conditions:**
- `LANGFUSE_ENABLED=true`
- `LANGFUSE_PUBLIC_KEY` is empty or missing
- OR `LANGFUSE_SECRET_KEY` is empty or missing

### 4. `unhealthy`
**Meaning:** Credentials are configured but authentication failed

**Response Example:**
```json
{
  "status": "unhealthy",
  "message": "Langfuse authentication failed"
}
```

**Conditions:**
- Credentials are present
- Langfuse package is installed
- Authentication check returns false (invalid credentials)

### 5. `error`
**Meaning:** An error occurred during the health check

**Response Example:**
```json
{
  "status": "error",
  "message": "Langfuse package not installed"
}
```

or

```json
{
  "status": "error",
  "message": "Langfuse connection error: [error details]"
}
```

**Conditions:**
- Langfuse package is not installed
- OR network connection error
- OR any other exception during the check

## Configuration

The endpoint uses the following environment variables:

- `LANGFUSE_ENABLED` - Master switch (default: `true`)
- `LANGFUSE_PUBLIC_KEY` - Langfuse public API key
- `LANGFUSE_SECRET_KEY` - Langfuse secret API key
- `LANGFUSE_HOST` - Langfuse server URL (default: `https://cloud.langfuse.com`)

## Testing

### Automated Tests
Run the test suite with:
```bash
python -m pytest backend/live_assist/api/test_langfuse_health.py -v
```

The test suite includes:
- Test with disabled Langfuse
- Test with missing public key
- Test with missing secret key
- Test with missing credentials (both)
- Test with invalid credentials
- Test response structure validation
- Test host field inclusion when healthy
- Configuration check method tests

### Manual Testing

#### Test 1: Disabled Langfuse
```bash
curl http://localhost:8000/health/langfuse
# Set LANGFUSE_ENABLED=false in .env first
```

#### Test 2: Missing Credentials
```bash
curl http://localhost:8000/health/langfuse
# Remove LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY from .env
```

#### Test 3: Invalid Credentials
```bash
curl http://localhost:8000/health/langfuse
# Set invalid keys in .env
```

#### Test 4: Valid Credentials
```bash
curl http://localhost:8000/health/langfuse
# Set valid Langfuse credentials in .env
```

## Implementation Details

### Location
- **Endpoint:** `backend/live_assist/api/app.py` (lines 48-111)
- **Tests:** `backend/live_assist/api/test_langfuse_health.py`

### Dependencies
- `langfuse` Python package
- FastAPI for the endpoint
- Configuration from `live_assist.core.config.Settings`

### Error Handling
The endpoint is designed to never crash the application:
- All exceptions are caught and returned as `error` status
- Missing credentials result in `misconfigured` status
- The system can operate normally regardless of health check result

## Design Decisions

1. **Graceful Degradation:** The endpoint never blocks or crashes the API, even if Langfuse is completely unavailable

2. **Clear Status Messages:** Five distinct statuses make it easy to diagnose configuration issues

3. **Security:** Credentials are never exposed in responses

4. **Simplicity:** No external dependencies beyond the langfuse package itself

## Related Documentation

- Main design document: `.kiro/specs/langfuse-observability/design.md`
- Requirements document: `.kiro/specs/langfuse-observability/requirements.md`
- Task list: `.kiro/specs/langfuse-observability/tasks.md` (Task 15)

## Future Enhancements

Potential improvements for future iterations:
- Add response time metrics
- Include Langfuse SDK version in response
- Add detailed connectivity diagnostics
- Include trace count or usage statistics
