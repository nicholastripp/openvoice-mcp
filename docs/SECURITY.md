# Security Guide

This guide covers the security features and best practices for the Home Assistant Realtime Voice Assistant.

## Overview

The voice assistant includes multiple layers of security to protect your home automation system and personal data:

- **Authentication**: Password-protected web UI with bcrypt hashing
- **Encryption**: HTTPS/TLS for all web communications
- **Authorization**: Session-based access control
- **Rate Limiting**: Protection against brute force attacks
- **CSRF Protection**: Prevention of cross-site request forgery
- **Security Headers**: Modern web security headers
- **File Permissions**: Secure storage of sensitive data

## Security Features

### 1. Authentication and Authorization

#### Password Protection
- Web UI requires authentication (username/password)
- Passwords are hashed using bcrypt with 12 rounds
- Password hashes stored in `.env` file, never in plaintext
- Sessions expire after configurable timeout (default: 1 hour)

#### Configuration
```yaml
web_ui:
  auth:
    enabled: true              # Enable authentication
    username: "admin"          # Default username
    session_timeout: 3600      # Session timeout in seconds
```

### 2. HTTPS/TLS Encryption

#### Automatic Certificate Generation
- Self-signed certificates generated automatically on first run
- Stored in `config/certs/` directory
- Valid for 365 days

#### Using Custom Certificates
```yaml
web_ui:
  tls:
    enabled: true
    cert_file: "config/certs/your-cert.pem"
    key_file: "config/certs/your-key.pem"
```

#### Let's Encrypt (Future)
Support for Let's Encrypt certificates is planned for a future release.

### 3. Rate Limiting

Protects against brute force attacks and abuse:

- **Authentication endpoints**: 5 attempts per minute
- **API endpoints**: 100 requests per minute
- **Configuration endpoints**: 20 requests per minute
- **Default rate**: 60 requests per minute

Rate limiting is implemented using a sliding window algorithm and returns appropriate `429 Too Many Requests` responses with `Retry-After` headers.

### 4. CSRF Protection

All state-changing operations require CSRF tokens:

- Double-submit cookie pattern
- Tokens validated on POST/PUT/DELETE requests
- Automatic token generation and validation
- Strict SameSite cookies

### 5. Security Headers

The following security headers are automatically applied:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
Content-Security-Policy: [see below]
```

#### Content Security Policy (CSP)
```
default-src 'self';
script-src 'self' 'unsafe-inline' 'unsafe-eval';
style-src 'self' 'unsafe-inline';
img-src 'self' data: https:;
font-src 'self';
connect-src 'self' wss: ws:;
frame-ancestors 'none';
base-uri 'self';
form-action 'self';
upgrade-insecure-requests
```

### 6. File Permissions

Sensitive files are automatically secured:

- `.env` file: `600` (read/write owner only)
- Config directory: `750` recommended
- Automatic permission checks on startup
- Warnings displayed for insecure permissions

To fix permissions manually:
```bash
chmod 600 .env
chmod 750 config
```

### 7. Request Size Limits

Prevents resource exhaustion attacks:

- Maximum request body size: 10MB
- Maximum URL length: 8KB
- Configurable limits per endpoint

## Best Practices

### 1. Initial Setup

1. **Run the installer** to set up authentication:
   ```bash
   ./install.sh
   ```

2. **Choose a strong password** when prompted (minimum 8 characters recommended)

3. **Store the password securely** - you'll need it to access the web UI

### 2. Network Security

#### Local Access Only
For maximum security, restrict access to localhost:
```yaml
web_ui:
  host: "127.0.0.1"  # Local access only
```

#### Firewall Rules
If allowing remote access, use firewall rules:
```bash
# Allow only from specific IP
sudo ufw allow from 192.168.1.100 to any port 8443

# Allow from local network only
sudo ufw allow from 192.168.1.0/24 to any port 8443
```

#### VPN Access
For remote access, consider using a VPN instead of exposing the service directly.

### 3. API Key Security

1. **Never commit API keys** to version control
2. **Use environment variables** for all sensitive data
3. **Rotate keys regularly**
4. **Use minimal permissions** for Home Assistant tokens

### 4. Regular Updates

1. **Keep dependencies updated**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Monitor security advisories** for dependencies

3. **Apply security patches** promptly

### 5. Monitoring and Logging

#### Security Event Logging
Monitor logs for security events:
```bash
grep -i "security\|auth\|rate limit" logs/assistant.log
```

#### Failed Login Attempts
```bash
grep "Authentication failed" logs/assistant.log
```

#### Rate Limit Violations
```bash
grep "Rate limit exceeded" logs/assistant.log
```

## Security Checklist

Before deploying to production:

- [ ] Strong password set for web UI
- [ ] HTTPS enabled (even with self-signed cert)
- [ ] `.env` file has 600 permissions
- [ ] Firewall configured for web UI port
- [ ] API keys use minimal required permissions
- [ ] Regular backups configured
- [ ] Log monitoring in place
- [ ] Update schedule established

## Incident Response

If you suspect a security breach:

1. **Immediately revoke** all API tokens and keys
2. **Change** the web UI password
3. **Review** logs for unauthorized access
4. **Generate** new Home Assistant access tokens
5. **Update** OpenAI API keys
6. **Audit** Home Assistant for any unauthorized changes

## Reporting Security Issues

If you discover a security vulnerability:

1. **Do not** open a public issue
2. **Email** security concerns to: [your-security-email]
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Future Security Enhancements

Planned security improvements include:

- **Multi-factor authentication** (TOTP/WebAuthn)
- **OAuth2/OIDC** integration
- **API token management** interface
- **Audit logging** with tamper protection
- **Encrypted configuration** storage
- **Certificate pinning** for API connections
- **Intrusion detection** system

## Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [Home Assistant Security](https://www.home-assistant.io/docs/authentication/)
- [OpenAI Security](https://openai.com/security)

---

Remember: Security is a continuous process, not a one-time setup. Regularly review and update your security configuration to maintain a strong security posture.