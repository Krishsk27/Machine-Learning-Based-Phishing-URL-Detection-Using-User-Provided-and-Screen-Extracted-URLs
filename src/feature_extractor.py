# src/feature_extractor.py
import re
from urllib.parse import urlparse
import ipaddress

def get_url_features(url):
    """
    Analyzes a URL and returns a list of numerical features.
    """
    features = []
    
    # Parse the URL
    try:
        parsed_url = urlparse(url)
        path = parsed_url.path
        hostname = parsed_url.netloc
    except:
        return None # Invalid URL

    # --- 1. Lexical Features (Counts) ---
    
    # Feature 1: Length of URL
    features.append(len(url))
    
    # Feature 2: Length of Hostname
    features.append(len(hostname))
    
    # Feature 3: Count of Dots (.) in hostname (Phishing often uses many subdomains)
    features.append(hostname.count('.'))
    
    # Feature 4: Count of Hyphens (-) in hostname (Phishing often uses 'secure-login-bank')
    features.append(hostname.count('-'))
    
    # Feature 5: Count of '@' (Used to obfuscate user credentials)
    features.append(url.count('@'))
    
    # Feature 6: Count of '?' (Query parameters)
    features.append(url.count('?'))
    
    # Feature 7: Count of '%' (Encoded characters)
    features.append(url.count('%'))

    # --- 2. Structural Features (Boolean: 1 = Yes, 0 = No) ---
    
    # Feature 8: Is it using IP Address instead of Domain? (e.g. http://192.168.1.5)
    try:
        ipaddress.ip_address(hostname)
        features.append(1) 
    except:
        features.append(0)

    # Feature 9: Is HTTPS used? (Phishers usually use HTTP, though this is changing)
    if parsed_url.scheme == 'https':
        features.append(0) # Safe
    else:
        features.append(1) # Risky (HTTP)

    # Feature 10: Deep Directory Depth? (e.g. .com/folder/folder/folder/login)
    features.append(path.count('/'))

    # Feature 11: Suspicious words in URL
    suspicious_keywords = ['login', 'verify', 'update', 'secure', 'account', 'banking']
    if any(word in url.lower() for word in suspicious_keywords):
        features.append(1)
    else:
        features.append(0)
        
    # Feature 12: Is it a Shortening Service? (bit.ly, tinyurl)
    shorteners = ['bit.ly', 'goo.gl', 'shorte.st', 'tinyurl.com', 'tr.im', 'is.gd', 'cli.gs']
    if any(s in hostname for s in shorteners):
        features.append(1)
    else:
        features.append(0)

    return features

# List of column names for the DataFrame later
feature_names = [
    'url_length', 'hostname_length', 'count_dots', 'count_hyphens', 
    'count_at', 'count_qmark', 'count_percent', 'is_ip', 
    'is_http', 'dir_depth', 'has_sus_words', 'is_shortened'
]