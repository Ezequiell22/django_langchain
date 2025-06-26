from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .agent_setup import sql_agent, vectorstore, analista_financeiro_chain, db
from django.core.cache import cache
import pandas as pd
from sqlalchemy import text

class PerguntaView(APIView):
    def post(self, request):
        pergunta = request.data.get("pergunta")
        if not pergunta:
            return Response({"erro": "Campo 'pergunta' é obrigatório."}, status=status.HTTP_400_BAD_REQUEST)

        cache_key = f"pergunta_cache:{pergunta.strip().lower()}"
        resposta_cacheada = cache.get(cache_key)
        if resposta_cacheada:
            return Response(resposta_cacheada | {"cache": True})

        try:
        
            contexto_docs = vectorstore.similarity_search(pergunta, k=1)
            contexto = " ".join(doc.page_content for doc in contexto_docs)

            
            prompt_sql = (
                f"Contexto do banco de dados Protheus: {contexto} "
                f"Gere apenas o SQL necessário, sem usar markdown, sem ```sql, "
                f"sem nenhum texto explicativo antes. Apenas a instrução SQL limpa. "
                f"Voce não deve executar o sql, apenas gerar o sql."
                f"Se não souber responder, diga: 'Desculpe, não sei.' "
                f"Pergunta: {pergunta}"
            )

           
            resposta_sql = sql_agent.invoke({"input": prompt_sql})
            sql_gerado = str(resposta_sql["output"]).strip()

            if "Desculpe" in sql_gerado:
                return Response({"resposta": sql_gerado, "sql": None, "cache": False})

            print("SQL gerado: ", sql_gerado)
           
            with db._engine.connect() as conn:
                result = conn.execute(text(sql_gerado))
                rows = result.fetchall()
                columns = result.keys()

            
            print("Dados: ", rows)
            df = pd.DataFrame(rows, columns=columns)
            tabela_formatada = df.to_markdown(index=False)

            
            resposta_analise = analista_financeiro_chain.invoke({
                "dados": tabela_formatada,
                "pergunta": pergunta
            })

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
