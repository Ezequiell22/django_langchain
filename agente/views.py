from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .agent_setup import sql_agent, vectorstore
import time
from django.core.cache import cache

class PerguntaView(APIView):
    def post(self, request):
        pergunta = request.data.get("pergunta")
        if not pergunta:
            return Response({"erro": "Campo 'pergunta' é obrigatório."}, status=status.HTTP_400_BAD_REQUEST)

        cache_key = f"pergunta_cache:{pergunta.strip().lower()}"
        resposta_cacheada = cache.get(cache_key)

        if resposta_cacheada:
            return Response({"resposta": resposta_cacheada, "cache": True})

        try:
            contexto_docs = vectorstore.similarity_search(pergunta, k=1)
            contexto = " ".join(doc.page_content for doc in contexto_docs)

            prompt_final = (
                f"Contexto do banco de dados Protheus: {contexto} "
                f"Responda sempre em português e gere o SQL necessário. "
                f"Se não souber responder, diga: 'Desculpe, não sei.' "
                f"Pergunta: {pergunta}"
            )

            resposta = sql_agent.invoke({"input": prompt_final})
            resposta_final = str(resposta["output"])

            cache.set(cache_key, resposta_final, timeout=3600)

            return Response({"resposta": resposta_final, "cache": False})
        except Exception as e:
            return Response({"erro": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
