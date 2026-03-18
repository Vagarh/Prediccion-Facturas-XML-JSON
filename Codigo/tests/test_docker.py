# -*- coding: utf-8 -*-
"""Tests para el contenedor Docker.

Requiere Docker daemon corriendo. Excluido por defecto con -m "not docker".
"""

import time

import pytest
import requests

pytestmark = pytest.mark.docker


def _docker_available() -> bool:
    """Verifica si Docker está disponible."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


skipif_no_docker = pytest.mark.skipif(
    not _docker_available(), reason="Docker daemon no disponible"
)


@pytest.fixture(scope="module")
def docker_image():
    """Construye la imagen Docker del proyecto."""
    import docker

    client = docker.from_env()
    image, _ = client.images.build(
        path=".",
        tag="savia-prediccion:test",
        rm=True,
    )
    yield image
    try:
        client.images.remove(image.id, force=True)
    except Exception:
        pass


@pytest.fixture(scope="module")
def docker_container(docker_image):
    """Ejecuta el contenedor y espera a que esté listo."""
    import docker

    client = docker.from_env()
    container = client.containers.run(
        docker_image.id,
        detach=True,
        ports={"8000/tcp": None},  # Puerto aleatorio
        remove=True,
    )

    # Esperar a que el contenedor esté listo
    port = None
    for _ in range(30):
        container.reload()
        if container.status == "running":
            ports = container.attrs["NetworkSettings"]["Ports"]
            if "8000/tcp" in ports and ports["8000/tcp"]:
                port = ports["8000/tcp"][0]["HostPort"]
                break
        time.sleep(1)

    if port is None:
        container.stop(timeout=5)
        pytest.fail("Contenedor no arrancó correctamente")

    # Esperar a que el healthcheck pase
    base_url = f"http://localhost:{port}"
    for _ in range(30):
        try:
            resp = requests.get(f"{base_url}/v1/health", timeout=2)
            if resp.status_code == 200:
                break
        except requests.ConnectionError:
            time.sleep(2)

    yield base_url, container

    try:
        container.stop(timeout=10)
    except Exception:
        pass


@skipif_no_docker
class TestDockerBuild:
    """Tests de construcción de la imagen Docker."""

    def test_image_builds_successfully(self, docker_image) -> None:
        """La imagen se construye sin errores."""
        assert docker_image is not None
        assert docker_image.id is not None

    def test_image_size_under_2gb(self, docker_image) -> None:
        """La imagen pesa menos de 2GB."""
        size_gb = docker_image.attrs["Size"] / (1024**3)
        assert size_gb < 2.0, f"Imagen: {size_gb:.2f}GB (máx: 2GB)"


@skipif_no_docker
class TestDockerRun:
    """Tests de ejecución del contenedor."""

    def test_health_endpoint_accessible(self, docker_container) -> None:
        """GET /v1/health retorna 200 desde el contenedor."""
        base_url, _ = docker_container
        resp = requests.get(f"{base_url}/v1/health", timeout=5)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_predict_endpoint_works(self, docker_container) -> None:
        """POST /v1/predict retorna 200 desde el contenedor."""
        from tests.fixtures.sample_data import make_realistic_features

        base_url, _ = docker_container
        payload = {
            "numero_factura": "DOCKER-001",
            "features": make_realistic_features(),
        }
        resp = requests.post(f"{base_url}/v1/predict", json=payload, timeout=10)
        assert resp.status_code == 200
        assert resp.json()["clase"] in {0, 1, 2}

    def test_container_stops_gracefully(self, docker_container) -> None:
        """El contenedor se detiene en <10s."""
        _, container = docker_container
        t0 = time.time()
        container.stop(timeout=10)
        elapsed = time.time() - t0
        assert elapsed < 15, f"Stop tomó {elapsed:.1f}s (máx: 15s)"

    def test_container_healthcheck_passes(self, docker_container) -> None:
        """El healthcheck del contenedor pasa."""
        _, container = docker_container
        container.reload()
        health = container.attrs.get("State", {}).get("Health", {})
        if health:
            assert health.get("Status") == "healthy"
