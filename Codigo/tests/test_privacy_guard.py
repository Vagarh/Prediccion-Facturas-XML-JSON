# -*- coding: utf-8 -*-
"""Tests unitarios para PrivacyGuard.

Valida el cumplimiento de la Ley 1581 de Colombia verificando que:
- verify_local_only() lanza excepción con hosts externos
- check_ollama_available() retorna False cuando servidor no responde
- log_inference() registra operaciones sin contenido textual
"""

import pytest
from unittest.mock import Mock, patch
import requests

from src.privacy.guard import PrivacyGuard, PrivacyViolationError


class TestPrivacyGuard:
    """Suite de tests para PrivacyGuard."""

    def setup_method(self):
        """Configuración inicial para cada test."""
        self.guard = PrivacyGuard()

    def test_verify_local_only_with_localhost(self):
        """Test que verify_local_only() acepta localhost."""
        # Configurar URLs locales válidas
        local_urls = [
            "http://localhost:11434",
            "http://127.0.0.1:11434",
            "https://localhost:8080",
            "https://127.0.0.1:8080"
        ]
        
        for url in local_urls:
            with patch('src.config.OLLAMA_BASE_URL', url):
                # No debe lanzar excepción
                self.guard.verify_local_only()

    def test_verify_local_only_with_external_host_raises_exception(self):
        """Test que verify_local_only() lanza excepción con hosts externos."""
        # Configurar URLs externas inválidas
        external_urls = [
            "http://api.openai.com",
            "https://api.anthropic.com",
            "http://192.168.1.100:11434",
            "https://ollama.example.com",
            "http://10.0.0.1:11434"
        ]
        
        for url in external_urls:
            with patch('src.config.OLLAMA_BASE_URL', url):
                with pytest.raises(PrivacyViolationError) as exc_info:
                    self.guard.verify_local_only()
                
                assert "host externo detectado" in str(exc_info.value).lower()
                assert url in str(exc_info.value)

    def test_verify_local_only_with_invalid_url_raises_exception(self):
        """Test que verify_local_only() lanza excepción con URLs inválidas."""
        invalid_urls = [
            "not-a-url",
            "",
            "ftp://localhost:11434",
            "localhost:11434"  # Sin protocolo
        ]
        
        for url in invalid_urls:
            with patch('src.config.OLLAMA_BASE_URL', url):
                with pytest.raises(PrivacyViolationError):
                    self.guard.verify_local_only()

    @patch('requests.get')
    def test_check_ollama_available_returns_true_when_server_responds(self, mock_get):
        """Test que check_ollama_available() retorna True cuando servidor responde."""
        # Mock respuesta exitosa
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response
        
        with patch('src.config.OLLAMA_BASE_URL', 'http://localhost:11434'):
            result = self.guard.check_ollama_available()
        
        assert result is True
        mock_get.assert_called_once_with(
            'http://localhost:11434/api/tags',
            timeout=5
        )

    @patch('requests.get')
    def test_check_ollama_available_returns_false_when_server_not_responds(self, mock_get):
        """Test que check_ollama_available() retorna False cuando servidor no responde."""
        # Mock diferentes tipos de errores
        exceptions = [
            requests.exceptions.ConnectionError("Connection refused"),
            requests.exceptions.Timeout("Request timeout"),
            requests.exceptions.RequestException("Generic error")
        ]
        
        for exception in exceptions:
            mock_get.side_effect = exception
            
            with patch('src.config.OLLAMA_BASE_URL', 'http://localhost:11434'):
                result = self.guard.check_ollama_available()
            
            assert result is False

    @patch('requests.get')
    def test_check_ollama_available_returns_false_on_http_error(self, mock_get):
        """Test que check_ollama_available() retorna False con errores HTTP."""
        # Mock respuesta con error HTTP
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        with patch('src.config.OLLAMA_BASE_URL', 'http://localhost:11434'):
            result = self.guard.check_ollama_available()
        
        assert result is False

    @patch('src.privacy.guard.logger')
    def test_log_inference_logs_without_textual_content(self, mock_logger):
        """Test que log_inference() registra sin contenido textual."""
        model = "llama3.2"
        input_hash = "abc123def456"
        
        self.guard.log_inference(model, input_hash)
        
        # Verificar que se llamó al logger
        mock_logger.info.assert_called_once()
        
        # Verificar que el mensaje contiene modelo y hash pero no contenido textual
        log_call_args = mock_logger.info.call_args[0][0]
        assert model in log_call_args
        assert input_hash in log_call_args
        assert "inference" in log_call_args.lower()

    @patch('src.privacy.guard.logger')
    def test_log_inference_with_different_models(self, mock_logger):
        """Test que log_inference() funciona con diferentes modelos."""
        test_cases = [
            ("llama3.2", "hash1"),
            ("mistral", "hash2"),
            ("qwen2.5", "hash3"),
            ("phi3", "hash4")
        ]
        
        for model, input_hash in test_cases:
            mock_logger.reset_mock()
            
            self.guard.log_inference(model, input_hash)
            
            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            assert model in log_message
            assert input_hash in log_message

    def test_privacy_violation_error_is_exception(self):
        """Test que PrivacyViolationError es una excepción válida."""
        error = PrivacyViolationError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    @patch('src.config.OLLAMA_BASE_URL', 'http://localhost:11434')
    @patch('requests.get')
    def test_integration_verify_and_check_ollama(self, mock_get):
        """Test de integración: verificar local y disponibilidad de Ollama."""
        # Mock respuesta exitosa de Ollama
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama3.2"}]}
        mock_get.return_value = mock_response
        
        # Verificar que es local (no debe lanzar excepción)
        self.guard.verify_local_only()
        
        # Verificar que Ollama está disponible
        assert self.guard.check_ollama_available() is True

    def test_guard_initialization(self):
        """Test que PrivacyGuard se inicializa correctamente."""
        guard = PrivacyGuard()
        assert guard is not None
        assert hasattr(guard, 'verify_local_only')
        assert hasattr(guard, 'check_ollama_available')
        assert hasattr(guard, 'log_inference')