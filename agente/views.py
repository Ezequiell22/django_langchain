from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .agent_setup import sql_agent, vectorstore, analista_financeiro_chain, db
from django.core.cache import cache
import pandas as pd
from sqlalchemy import text
import time
import hashlib

class PerguntaView(APIView):
    def post(self, request):
        pergunta = request.data.get("pergunta")
        if not pergunta:
            return Response({"erro": "Campo 'pergunta' é obrigatório."}, status=status.HTTP_400_BAD_REQUEST)

        cache_key = f"pergunta_cache:{pergunta.strip().lower()}"
        cache_key = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
        resposta_cacheada = cache.get(cache_key)
        if resposta_cacheada:
            return Response(resposta_cacheada | {"cache": True})

        try:
            start = time.time()
            contexto_docs = vectorstore.similarity_search(pergunta, k=1)
            contexto = " ".join(doc.page_content for doc in contexto_docs)
            print("Tempo busca no vectorstore:", round(time.time() - start, 2), "s")
            
            prompt_sql = (
                f"Contexto do banco de dados Protheus: {contexto} "
                f"Gere apenas o SQL necessário, sem usar markdown, sem ```sql, "
                f"sem nenhum texto explicativo antes. Apenas a instrução SQL limpa. "
                f"Voce não deve executar o sql, apenas gerar o sql."
                f"Se não souber responder, diga: 'Desculpe, não sei.' "
                f"Pergunta: {pergunta}"
            )

            t2 = time.time()
            resposta_sql = sql_agent.invoke({"input": prompt_sql})
            sql_gerado = str(resposta_sql["output"]).strip()
            print("Tempo de geração do script SQL:", round(time.time() - t2, 2), "s")

            if "Desculpe" in sql_gerado:
                return Response({"resposta": sql_gerado, "sql": None, "cache": False})

            print("SQL gerado: ", sql_gerado)
           
            t3 = time.time()
            with db._engine.connect() as conn:
                result = conn.execute(text(sql_gerado))
                rows = result.fetchall()
                columns = result.keys()
            print("Tempo de consulta SQL:", round(time.time() - t3, 2), "s")

            
            print("Dados obtidos na consulta: ", rows)
            df = pd.DataFrame(rows, columns=columns)
            tabela_formatada = df.to_markdown(index=False)

            t4 = time.time()
            resposta_analise = analista_financeiro_chain.invoke({
                "dados": tabela_formatada,
                "pergunta": pergunta
            })
            print("Tempo processamento do agente financeiro:", round(time.time() - t4, 2), "s")

            resultado_final = {
                "resposta_sql": sql_gerado,
                "tabela": tabela_formatada,
                "relatorio": resposta_analise["text"],
                "cache": False
            }

            cache.set(cache_key, resultado_final, timeout=3600)
            return Response(resultado_final)

        except Exception as e:
            return Response({"erro": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
