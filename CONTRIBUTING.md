# Como contribuir com o Ocarina Recomp

[Read in English](CONTRIBUTING.en.md)

Toda ajuda conta, inclusive sem saber programar. Este guia explica como testar, como compilar, como investigar um bug gráfico e como enviar sua contribuição.

## A regra mais importante

**Nada que venha do jogo entra neste repositório.** Isso inclui:

- o jogo, keys, `exefs`, `romfs` e o `.nup`;
- o `ocarina.exe`, o código em `generated/` e tudo em `build/`;
- saves, cache de shaders, SPIR-V dos shaders do jogo, dumps de frame e prints.

Hashes, endereços (`main+0x6e6dc4`) e a descrição do que o código faz podem entrar.

O `tools/check_content.py` confere isso em todo push. Ative a verificação local uma vez por clone:

```powershell
git config core.hooksPath .githooks
```

Nunca use `--no-verify`. E não peça nem compartilhe o jogo em issues ou pull requests.

## Sem programar: testar

Compile pelo [passo a passo](docs/compilar-manual.md), jogue e abra uma [issue](https://github.com/sostenesfreitas/ocarina-recomp/issues) contando:

- placa de vídeo, versão do driver, processador e quantidade de RAM;
- onde no jogo aconteceu e como repetir;
- o que você esperava ver e o que apareceu;
- o `ultima-partida-erros.log` da pasta de instalação.

**O mais útil agora:** testes em placas AMD e Intel, em PCs mais fracos e em áreas além do vilarejo Kokiri.

## Preparar o ambiente

Siga o [passo a passo para compilar](docs/compilar-manual.md): ele deixa as ferramentas, o `.venv` e o `build/` prontos em `C:\OcarinaRecomp`. Depois, faça seu fork e aponte o `origin` para ele:

```bash
git remote rename origin upstream
git remote add origin https://github.com/SEU-USUARIO/ocarina-recomp
```

Para recompilar, use um terminal com `C:\msys64\ucrt64\bin` no `PATH`:

```bash
# O runtime mudou: é incremental e leva pouco tempo
.venv/Scripts/python tools/build_game.py --version oot111 --gen generated/oot111 --out build/oot111 \
  --expected tools/data/expected-oot111.json --exefs build/exefs --romfs build/romfs

# O tradutor de shaders mudou: rode isto e depois o comando acima
.venv/Scripts/python tools/build_xlate.py

# O lifter (recomp/) mudou: gere de novo e depois rode o primeiro comando
.venv/Scripts/python recomp/gen.py --exefs build/exefs --out generated/oot111 --modules main,sdk,subsdk0
.venv/Scripts/python tools/oot/patch_generated.py generated/oot111
```

Para abrir o jogo, use `jogar\jogar-ocarina.ps1`.

## Onde fica cada coisa

| Pasta | Conteúdo |
|---|---|
| `recomp/` | Tradutor de ARM64 para C |
| `runtime/` | O "console": kernel, sistema, áudio, entrada e NVN sobre Vulkan |
| `runtime/vk_exec_draw.c` | Execução de draws e dispatches, inclusive mesh |
| `runtime/vk_pipeline.c` | Criação de pipelines |
| `runtime/ov_*.c` | Substituições de funções do SDK (`HOST_OVERRIDE`) |
| `runtime/third_party/eglantine/.../ampere/` | Rebaixamento de shaders Ampere para Maxwell |
| `tools/xlate/patches/` | Patches do tradutor de shaders. A quantidade esperada está em `tools/build_xlate.py` |
| `tools/oot/` | Partes específicas do Ocarina |
| `instalador/`, `tools/oot/make_setup.py` | O instalador com janela e o script que monta o ZIP |
| `jogar/` | O lançador e a configuração inicial |
| `verificar.ps1`, `compilar.ps1` | Os scripts de compilação. Ao mudar um comando, mude também o instalador e os guias em `docs/` |

Os prefixos `TOTK_` e `DKB_` nos nomes vêm dos projetos de origem. Eles continuam assim para o código seguir fácil de comparar com esses projetos. Comentários que citam outros jogos registram onde o comportamento foi descoberto.

## Regras do código

- **Não edite o código gerado à mão.** Ele é recriado. Use `recomp/patches.py` ou, se não couber, `tools/oot/patch_generated.py`.
- **Marcador `LOCAL TEST`:** mudanças feitas durante a investigação que ainda não têm forma definitiva. Ao melhorar um trecho desses, tire o marcador.
- **Patch do tradutor:** gere com um diff sobre `build/xlate/ryujinx` e atualize a contagem em `tools/build_xlate.py`. Mudar o tradutor recompila todos os shaders na próxima execução.
- **Windows:** `long` tem 32 bits. Foi um `1L << 40` que descartou todos os draws indiretos de mesh por dias.
- **Código copiado:** mantenha o cabeçalho e adicione uma linha em `THIRD_PARTY_NOTICES.md`, com o commit de origem.
- **Commits:** use a sua própria identidade do git. Nunca envie direto para `main`: abra um pull request.

## Como investigar um bug gráfico

O que resolveu os bugs maiores até agora foi **medir e comparar com uma referência**, sem apostar em palpite.

1. **Crie um cenário repetível.** Use um script de entrada (`TOTK_INPUT_SCRIPT`) com `TOTK_INPUT=none`, porque senão o seu teclado mexe no Link enquanto a janela de teste está em foco:
   ```text
   700 A 8
   2300 RSTICK 20000 0 150
   2600 LSTICK 0 32000 240
   ```
2. **Capture.** `ocarina.exe --seconds 120 --shot saida.png`. Com `TOTK_SHOTS=lo-hi:passo`, ele salva miniaturas ao longo do caminho.
3. **Isole.** Desligue um programa de cada vez até o defeito sumir, por exemplo com `OOT_MESH_DENY=4049,4047`.
4. **Compare com o Lichii.** Com `EDEN_NX2_RENDER_TRACE=1`, ele registra os draws, e dá para comparar com os nossos.
5. **Leia o shader.** Use `TOTK_DUMP=<frame>` e `TOTK_DUMP_DIR=<pasta>`, descompile com `spirv-cross` e compare com o código nativo. Um valor absurdo costuma apontar a causa: foi assim que achamos a colisão de bancos de registradores.

### Ferramentas de diagnóstico

| Variável | O que faz |
|---|---|
| `OOT_DRAW_ARGS=<frame>` | Registra tipos de draw, argumentos, buffers por estágio e estatísticas por programa |
| `OOT_MESH_DENY=<a,b,...>` | Descarta draws indiretos de mesh desses programas |
| `OOT_NO_CULL=1`, `OOT_NO_CULL_PROGS=lo-hi` | Desliga o descarte de faces |
| `OOT_FOLLOW_DISP=<frame>:<k>:<c0>` | Segue um dispatch, junto com `TOTK_DUMP_BUF=0:<tamanho>` |
| `TOTK_XLATE_IR=<pasta>` | O tradutor grava os blocos e o IR de cada shader com subrotina |
| `TOTK_SPV_OVERRIDE=<pasta>` | Troca shaders por arquivos `sh_<hash>.spv`. O hash muda quando o tradutor muda, e o arquivo antigo para de valer sem aviso |
| `OOT_BIG_CS_WORDS`, `OOT_BIG_CS_OPT=1`, `OOT_BIG_CS_THREADS`, `OOT_BIG_CS_HUGE` | Controlam a compilação em segundo plano dos compute shaders grandes: o tamanho a partir do qual vão para o segundo plano, a otimização, quantos compilam juntos (padrão 2 com 24 GB de RAM ou mais, senão 1) e a partir de que tamanho só um por vez (padrão 262144 palavras). O menor da fila vai primeiro |
| `TOTK_TRACE=perf` | Registra o tráfego de memória por frame. Comece por aqui quando o fps estiver baixo |

**Ao ler buffers da GPU:** leia depois do `VK_MEM_USE`. Antes dele, você vê uma cópia antiga da CPU.

## Bons pontos para começar

- Por que a segunda passada por uma área ainda monta pipelines novos.
- Testar e corrigir em AMD e Intel.
- Conferir a porta da casa do Link, quebrar vasos e tocar a ocarina.
- Conferir o fog em Hyrule Field.
- Conferir o filme de abertura de New Game.
- Avaliar os PRs #224, #353, #337, #215 e #345 do Lichii.
- Trocar a lista de candidatos de ponteiros pela tabela bruta de pools do nxnative (`TryGenerateRawStorageOp`), o que reduziria os compute shaders gigantes.

## Enviar um pull request

1. Crie um branch a partir de `main`.
2. Faça uma mudança por pull request.
3. Rode `python tools/check_content.py`. Se mexeu no lifter, rode também os testes em `tests/`.
4. Teste no jogo. Diga no pull request qual placa e driver usou, em que cena e o que mudou. Se puder medir, melhor ainda: "fps de X para Y", ou "programa Z passou de 0 para N triângulos".
5. Diga o que você **não** testou. Um "não testei em AMD" evita muita surpresa.

Ao usar IA para ajudar, revise o resultado e confirme que funciona antes de enviar. Veja também o [código de conduta](CODE_OF_CONDUCT.md).
