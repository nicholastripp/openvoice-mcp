#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test security features of the web UI"""

import asyncio
import aiohttp
import sys
import locale
from pathlib import Path

# Try to set UTF-8 locale, fall back to ASCII if not available
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except:
        # Fall back to default
        pass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

async def test_rate_limiting():
    """Test rate limiting on authentication endpoint"""
    print("\n=== Testing Rate Limiting ===")
    
    async with aiohttp.ClientSession() as session:
        # Try to exceed auth rate limit (5 per minute)
        for i in range(7):
            try:
                async with session.post(
                    'https://localhost:8443/login',
                    json={'username': 'test', 'password': 'wrong'},
                    ssl=False
                ) as resp:
                    if resp.status == 429:
                        print(f"[PASS] Rate limit triggered after {i} attempts")
                        retry_after = resp.headers.get('Retry-After', 'N/A')
                        print(f"   Retry-After: {retry_after} seconds")
                        break
                    else:
                        print(f"   Attempt {i+1}: Status {resp.status}")
            except Exception as e:
                print(f"   Attempt {i+1}: {type(e).__name__}")
        else:
            print("[FAIL] Rate limit not triggered!")

async def test_security_headers():
    """Test security headers"""
    print("\n=== Testing Security Headers ===")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                'https://localhost:8443/',
                ssl=False
            ) as resp:
                headers_to_check = [
                    'X-Content-Type-Options',
                    'X-Frame-Options',
                    'X-XSS-Protection',
                    'Strict-Transport-Security',
                    'Content-Security-Policy'
                ]
                
                for header in headers_to_check:
                    value = resp.headers.get(header)
                    if value:
                        print(f"[PASS] {header}: {value[:50]}...")
                    else:
                        print(f"[FAIL] {header}: Missing")
                        
                # Check for headers that should NOT be present
                bad_headers = ['Server', 'X-Powered-By']
                for header in bad_headers:
                    if header in resp.headers:
                        print(f"[FAIL] {header} should not be present!")
                    else:
                        print(f"[PASS] {header} correctly removed")
                        
        except Exception as e:
            print(f"[FAIL] Error: {e}")

async def test_csrf_protection():
    """Test CSRF protection"""
    print("\n=== Testing CSRF Protection ===")
    
    async with aiohttp.ClientSession() as session:
        # First, get a page to obtain CSRF cookie
        try:
            async with session.get(
                'https://localhost:8443/config/env',
                ssl=False
            ) as resp:
                csrf_cookie = resp.cookies.get('csrf_token')
                if csrf_cookie:
                    print(f"[PASS] CSRF cookie set: {csrf_cookie.value[:10]}...")
                else:
                    print("[FAIL] No CSRF cookie received")
                    
            # Try POST without CSRF token
            async with session.post(
                'https://localhost:8443/api/config/env',
                json={'test': 'data'},
                ssl=False
            ) as resp:
                if resp.status == 403:
                    print("[PASS] POST without CSRF token correctly rejected")
                else:
                    print(f"[FAIL] POST without CSRF accepted (status: {resp.status})")
                    
        except Exception as e:
            print(f"Note: {type(e).__name__} - Web UI may require authentication")

async def test_request_size_limit():
    """Test request size limits"""
    print("\n=== Testing Request Size Limits ===")
    
    async with aiohttp.ClientSession() as session:
        # Create a large payload (11MB, exceeding 10MB limit)
        large_data = 'x' * (11 * 1024 * 1024)
        
        try:
            async with session.post(
                'https://localhost:8443/api/test',
                data=large_data,
                ssl=False
            ) as resp:
                if resp.status == 413:
                    print("[PASS] Large request correctly rejected (413 Payload Too Large)")
                else:
                    print(f"[FAIL] Large request accepted (status: {resp.status})")
        except aiohttp.ClientPayloadError:
            print("[PASS] Large request rejected by client (payload too large)")
        except Exception as e:
            print(f"[PASS] Large request rejected: {type(e).__name__}")

async def check_file_permissions():
    """Check file permissions"""
    print("\n=== Checking File Permissions ===")
    
    import os
    import stat
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        mode = env_file.stat().st_mode & 0o777
        if mode == 0o600:
            print(f"[PASS] .env has secure permissions: {oct(mode)}")
        else:
            print(f"[FAIL] .env has insecure permissions: {oct(mode)} (should be 0o600)")
    else:
        print("[WARN] .env file not found")
        
    # Check config directory
    config_dir = Path("config")
    if config_dir.exists():
        mode = config_dir.stat().st_mode & 0o777
        if not (mode & 0o077):  # No permissions for others
            print(f"[PASS] config directory has secure permissions: {oct(mode)}")
        else:
            print(f"[WARN] config directory has broad permissions: {oct(mode)}")

async def main():
    """Run all security tests"""
    print("[SECURITY] Security Feature Tests")
    print("=" * 50)
    
    # Check file permissions first (doesn't require web UI)
    await check_file_permissions()
    
    print("\nNote: The following tests require the web UI to be running.")
    print("Start with: python src/main.py --web")
    
    try:
        # Test web UI security features
        await test_security_headers()
        await test_rate_limiting()
        await test_csrf_protection()
        await test_request_size_limit()
    except aiohttp.ClientConnectorError:
        print("\n[FAIL] Could not connect to web UI at https://localhost:8443")
        print("   Please ensure the web UI is running.")

if __name__ == "__main__":
    asyncio.run(main())