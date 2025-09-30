# SSL Certificate Configuration

OptILLM now supports SSL certificate verification configuration to work with self-signed certificates or corporate proxies.

## Usage

### Disable SSL Verification (Development Only)

**⚠️ WARNING: Only use this in development environments. Disabling SSL verification is insecure.**

#### Via Command Line
```bash
python optillm.py --no-ssl-verify
```

#### Via Environment Variable
```bash
export OPTILLM_SSL_VERIFY=false
python optillm.py
```

### Use Custom CA Certificate Bundle

For corporate environments with custom Certificate Authorities:

#### Via Command Line
```bash
python optillm.py --ssl-cert-path /path/to/ca-bundle.crt
```

#### Via Environment Variable
```bash
export OPTILLM_SSL_CERT_PATH=/path/to/ca-bundle.crt
python optillm.py
```

## Configuration Options

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `--ssl-verify` / `--no-ssl-verify` | `OPTILLM_SSL_VERIFY` | `true` | Enable/disable SSL certificate verification |
| `--ssl-cert-path` | `OPTILLM_SSL_CERT_PATH` | `""` | Path to custom CA certificate bundle |

## Affected Components

SSL configuration applies to:
- **OpenAI API clients** (OpenAI, Azure, Cerebras)
- **HTTP plugins** (readurls, deep_research)
- **All external HTTPS connections**

## Examples

### Development with Self-Signed Certificate
```bash
# Disable SSL verification temporarily
python optillm.py --no-ssl-verify --base-url https://localhost:8443/v1
```

### Production with Corporate CA
```bash
# Use corporate certificate bundle
python optillm.py --ssl-cert-path /etc/ssl/certs/corporate-ca-bundle.crt
```

### Docker Environment
```bash
docker run -e OPTILLM_SSL_VERIFY=false optillm
```

## Security Notes

1. **Never disable SSL verification in production** - This makes your application vulnerable to man-in-the-middle attacks
2. **Use custom CA bundles instead** - For corporate environments, provide the proper CA certificate path
3. **Warning messages** - When SSL verification is disabled, OptILLM will log a warning message for security awareness

## Testing

Run the SSL configuration test suite:
```bash
python -m unittest tests.test_ssl_config -v
```

This validates:
- CLI argument parsing
- Environment variable configuration
- HTTP client SSL settings
- Plugin SSL propagation
- Warning messages