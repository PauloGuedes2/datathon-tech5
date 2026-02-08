"""
Controlador de monitoramento da API.

Responsabilidades:
- Expor endpoint de dashboard de monitoramento
- Fornecer dependência do serviço de monitoramento
"""

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from src.application.monitoring_service import ServicoMonitoramento


def obter_servico_monitoramento():
    """
    Dependência para obter uma instância do serviço de monitoramento.

    Retorno:
    - ServicoMonitoramento: instância pronta para uso
    """
    return ServicoMonitoramento()


class ControladorMonitoramento:
    """
    Controlador para endpoints de monitoramento.

    Responsabilidades:
    - Registrar rota do dashboard
    - Retornar HTML gerado pelo Evidently
    """

    def __init__(self):
        """
        Inicializa o controlador.

        Responsabilidades:
        - Criar o roteador
        - Registrar a rota do dashboard
        """
        self.roteador = APIRouter()
        self.roteador.add_api_route(
            "/dashboard",
            self._obter_dashboard,
            methods=["GET"],
            response_class=HTMLResponse,
        )

    @staticmethod
    async def _obter_dashboard(servico: ServicoMonitoramento = Depends(obter_servico_monitoramento)):
        """
        Retorna o dashboard do Evidently AI.

        Parâmetros:
        - servico (ServicoMonitoramento): serviço de monitoramento

        Retorno:
        - HTMLResponse: HTML do dashboard
        """
        conteudo_html = servico.gerar_dashboard()
        return HTMLResponse(content=conteudo_html, status_code=200)
