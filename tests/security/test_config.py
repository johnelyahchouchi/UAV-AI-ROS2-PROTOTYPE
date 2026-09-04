import pytest

from uav_security.config import SecurityConfigurationError, SecurityLimits
from uav_security.transport import client_tls_files, server_tls_files


@pytest.mark.parametrize(
    "environment",
    [
        {"UAV_MAX_HEADER_SIZE": "0"},
        {"UAV_MAX_JPEG_SIZE": str(1024 * 1024 * 1024)},
        {"UAV_SOCKET_READ_TIMEOUT": "nan"},
        {"UAV_MAX_DETECTIONS": "not-an-integer"},
    ],
)
def test_invalid_security_limit_overrides_fail_closed(environment):
    with pytest.raises(SecurityConfigurationError):
        SecurityLimits.from_environment(environment)


def test_missing_server_tls_material_has_no_plaintext_fallback():
    with pytest.raises(SecurityConfigurationError, match="required"):
        server_tls_files({})


def test_missing_client_tls_material_has_no_plaintext_fallback():
    with pytest.raises(SecurityConfigurationError, match="required"):
        client_tls_files({})
