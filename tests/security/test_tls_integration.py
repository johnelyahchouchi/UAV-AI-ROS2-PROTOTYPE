from __future__ import annotations

from datetime import datetime, timedelta, timezone
import ipaddress
from pathlib import Path
import socket
import ssl
import threading

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from uav_security.transport import (
    TLSFiles,
    connect_tls_sender,
    create_client_tls_context,
    create_server_tls_context,
    encode_packet,
    receive_packet,
    validate_negotiated_tls,
)


def issue_certificate(
    directory: Path,
    name: str,
    *,
    ca_key,
    ca_certificate,
    usage,
    dns_name: str | None = None,
) -> tuple[Path, Path]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = datetime.now(timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_certificate.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([usage]), critical=False)
    )
    if dns_name:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(dns_name)]), critical=False
        )
    certificate = builder.sign(ca_key, hashes.SHA256())
    cert_path = directory / f"{name}.crt"
    key_path = directory / f"{name}.key"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


def create_ca(directory: Path, name: str):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=None,
                decipher_only=None,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    path = directory / f"{name}.crt"
    path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return key, certificate, path


def tls_material(tmp_path: Path):
    ca_key, ca_cert, ca_path = create_ca(tmp_path, "trusted-ca")
    server_cert, server_key = issue_certificate(
        tmp_path,
        "server",
        ca_key=ca_key,
        ca_certificate=ca_cert,
        usage=ExtendedKeyUsageOID.SERVER_AUTH,
        dns_name="localhost",
    )
    client_cert, client_key = issue_certificate(
        tmp_path,
        "client",
        ca_key=ca_key,
        ca_certificate=ca_cert,
        usage=ExtendedKeyUsageOID.CLIENT_AUTH,
    )
    other_key, other_ca, _ = create_ca(tmp_path, "untrusted-ca")
    bad_cert, bad_key = issue_certificate(
        tmp_path,
        "untrusted-client",
        ca_key=other_key,
        ca_certificate=other_ca,
        usage=ExtendedKeyUsageOID.CLIENT_AUTH,
    )
    return (
        TLSFiles(server_cert, server_key, ca_path),
        TLSFiles(client_cert, client_key, ca_path),
        TLSFiles(bad_cert, bad_key, ca_path),
    )


def start_server(context: ssl.SSLContext, result: dict):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def run():
        try:
            raw, _ = listener.accept()
            with context.wrap_socket(raw, server_side=True) as connection:
                validate_negotiated_tls(connection)
                result["packet"] = receive_packet(connection)
                connection.sendall(b"ok")
        except Exception as error:
            result["error"] = error
        finally:
            listener.close()

    thread = threading.Thread(target=run)
    thread.start()
    return listener.getsockname()[1], thread


def test_valid_mutually_authenticated_tls_session(tmp_path: Path, valid_header):
    server_files, client_files, _ = tls_material(tmp_path)
    result = {}
    port, thread = start_server(create_server_tls_context(server_files), result)
    connection = connect_tls_sender(
        "127.0.0.1",
        port,
        context=create_client_tls_context(client_files),
        server_name="localhost",
    )
    with connection:
        connection.sendall(encode_packet(valid_header, b"jpeg"))
        assert connection.recv(2) == b"ok"
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert "error" not in result
    assert result["packet"].jpeg == b"jpeg"


def test_untrusted_client_certificate_is_rejected(tmp_path: Path):
    server_files, _, bad_client_files = tls_material(tmp_path)
    result = {}
    port, thread = start_server(create_server_tls_context(server_files), result)
    try:
        connection = connect_tls_sender(
            "127.0.0.1",
            port,
            context=create_client_tls_context(bad_client_files),
            server_name="localhost",
        )
        with connection:
            connection.sendall(b"untrusted")
    except (ssl.SSLError, ConnectionError):
        pass
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert isinstance(result.get("error"), ssl.SSLError)
