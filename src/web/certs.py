"""
TLS certificate management for the web UI
"""
import logging
import os
import ssl
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def create_self_signed_cert(cert_dir: Path, hostname: str = "localhost") -> Tuple[Path, Path]:
    """
    Create a self-signed certificate for HTTPS.
    Returns paths to (cert_file, key_file).
    """
    cert_dir.mkdir(parents=True, exist_ok=True)
    
    cert_file = cert_dir / "self-signed.crt"
    key_file = cert_dir / "self-signed.key"
    
    # Check if certificate already exists
    if cert_file.exists() and key_file.exists():
        logger.info("Self-signed certificate already exists")
        return cert_file, key_file
    
    logger.info("Generating self-signed certificate...")
    
    try:
        # Use openssl to generate certificate
        # This avoids the cryptography dependency
        cmd = [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key_file),
            "-out", str(cert_file),
            "-days", "365",
            "-nodes",  # No password
            "-subj", f"/CN={hostname}/O=HA Voice Assistant/C=US"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"OpenSSL failed: {result.stderr}")
            
        # Set proper permissions
        os.chmod(key_file, 0o600)  # Private key readable only by owner
        os.chmod(cert_file, 0o644)  # Certificate readable by all
        
        logger.info(f"Generated self-signed certificate: {cert_file}")
        return cert_file, key_file
        
    except FileNotFoundError:
        logger.error("OpenSSL not found. Please install OpenSSL to generate certificates.")
        raise
    except Exception as e:
        logger.error(f"Failed to generate certificate: {e}")
        raise


def create_ssl_context(cert_file: Path, key_file: Path) -> ssl.SSLContext:
    """Create SSL context for HTTPS server"""
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(str(cert_file), str(key_file))
    
    # Set secure defaults
    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
    ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
    
    return ssl_context


def get_certificate_info(cert_file: Path) -> dict:
    """Get information about a certificate"""
    try:
        # Use openssl to get certificate info
        cmd = ["openssl", "x509", "-in", str(cert_file), "-noout", "-text"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return {"error": "Failed to read certificate"}
            
        # Parse basic info
        info = {
            "file": str(cert_file),
            "exists": cert_file.exists()
        }
        
        # Extract validity dates
        cmd_dates = ["openssl", "x509", "-in", str(cert_file), "-noout", "-dates"]
        result_dates = subprocess.run(cmd_dates, capture_output=True, text=True)
        
        if result_dates.returncode == 0:
            for line in result_dates.stdout.splitlines():
                if line.startswith("notBefore="):
                    info["valid_from"] = line.split("=", 1)[1]
                elif line.startswith("notAfter="):
                    info["valid_until"] = line.split("=", 1)[1]
                    
        return info
        
    except Exception as e:
        logger.error(f"Failed to get certificate info: {e}")
        return {"error": str(e)}


def validate_cert_key_pair(cert_file: Path, key_file: Path) -> bool:
    """Validate that a certificate and key file match"""
    try:
        # Get modulus of certificate
        cmd_cert = ["openssl", "x509", "-noout", "-modulus", "-in", str(cert_file)]
        result_cert = subprocess.run(cmd_cert, capture_output=True, text=True)
        
        # Get modulus of key
        cmd_key = ["openssl", "rsa", "-noout", "-modulus", "-in", str(key_file)]
        result_key = subprocess.run(cmd_key, capture_output=True, text=True)
        
        if result_cert.returncode != 0 or result_key.returncode != 0:
            return False
            
        # Compare modulus values
        return result_cert.stdout.strip() == result_key.stdout.strip()
        
    except Exception as e:
        logger.error(f"Failed to validate cert/key pair: {e}")
        return False