# Desempenho e imagem no Linux com NVIDIA

O que mudou neste fork para o gameplay passar de ~12 fps para ~50 fps numa RTX 3070 (Linux, driver 580), com a
imagem igual à do Eglantine. Também lista o que ainda falta para 60 fps e as ferramentas usadas para medir.

Cenário de todas as medições: Floresta Kokiri, na varanda da casa do Link, save carregado direto e lista de
`DKB_SKIP_CS` padrão. O brilho ("luma") é a média de cinza da tela, de 0 a 255. O Eglantine mostra 58 nesse ponto.

## Resultado

| Etapa | fps | Brilho | Commit |
|---|---|---|---|
| Início (modo shadow) | 12,5 | — | — |
| Cópias redundantes do shadow | 24 | — | `7cca680` |
| RomFS da atualização 1.1.1 (cores certas) | ~55, mas a vegetação parada | — | `f0facce` |
| Todos os computes grandes compilados (vegetação e folhas se mexem) | 25 | 71 | `27bf943` |
| Mirror sem re-upload, merge até 16 MiB, `lag3`, EV 0 | ~50 | 60 | `3f0eedd` |
| Replay mais leve (15 → 13 ms por frame) | ~50 | 60 | `a89bbd3` |
| Captura das escritas da CPU no submit (`OOT_CAPTURE`) | ~55-57 | 61 | este |

Com o monitor a 100 Hz, 50 fps é exatamente uma imagem a cada 2 atualizações da tela, um ritmo regular.

## Mudanças

### Cores: a RomFS precisa ser a da atualização (`f0facce`)

O executável é o da versão 1.1.1, mas a RomFS tinha sido montada a partir do pacote base (1.0) do `zelda.nup`. Os
shaders da 1.0 liam as constantes da 1.1.1 nos lugares errados: imagem laranja e roxa, sem vegetação e 3D preto com a
lista curta de `DKB_SKIP_CS`. A correção é sobrepor os 2.280 arquivos que a atualização muda (pasta `romfs-patches`).
O runtime agora avisa quando a RomFS é a da 1.0 ("RomFS: WARNING: this RomFS is the base game's"), e os guias de
compilação mandam usar `--overlay-update`.

### Computes grandes: o menor primeiro (`27bf943`)

Shaders de compute muito grandes são compilados em segundo plano, e os dispatches deles são pulados até ficarem
prontos. A fila era em ordem de chegada, então os de vegetação esperavam atrás dos de 337 mil words da rede de
upscale do vídeo. Por isso as folhas e a vegetação ficavam paradas depois de carregar o save. Agora a fila compila o
menor primeiro (`TOTK_BIG_CS_FIFO=1` volta à ordem antiga). `ocarina --warm-big-cs` deixa todos no
`pipeline-cache.bin` de uma vez (cerca de 30 minutos na primeira vez).

### Mirror: o write-back não volta para a VRAM (`3f0eedd`)

No modo mirror (o padrão na NVIDIA), o que um shader escreve vai para uma cópia na VRAM e volta para a memória do
jogo no fim do submit. Ao voltar, as páginas eram marcadas como escritas, e o sync seguinte mandava os mesmos bytes
de novo para a VRAM: 72 MiB por frame. Agora `vk_gpu_written_mirrored` marca essas páginas, e o sync as pula enquanto
a CPU não as reescrever. `TOTK_MIRROR_WB_RESYNC=1` volta ao comportamento antigo.

### Merge até 16 MiB (`3f0eedd`)

Bindings guardados de 8 MiB voltavam por cópia inteira: 45 MiB por frame pela PCIe, por cima dos bytes que a CPU tinha
escrito enquanto isso. Agora voltam pelo merge, que só copia as palavras que a GPU mudou. O limite é
`OOT_MERGE_MAX_MIB` (padrão 16, antes 4).

### GPU até 3 submits atrás e EV 0 (`3f0eedd`)

`OOT_SUBMIT_SYNC` fazia cada submit do jogo esperar a GPU terminar, então CPU e GPU nunca trabalhavam ao mesmo tempo.
Isso também deixava a imagem lavada (brilho 115). O hack de exposição do upstream (`OOT_EV=-1.25`) compensava esse
efeito, mas só escurecia tudo por igual, sem recuperar a névoa e o contraste.

Agora o padrão é `lag3`: o jogo espera a GPU do submit de 3 atrás. A floresta fica igual ao Eglantine sem correção
nenhuma (brilho 60), então o padrão de `OOT_EV` passou a ser 0.

| `OOT_SUBMIT_SYNC` | Efeito |
|---|---|
| `1` | Espera a GPU do próprio submit (o comportamento antigo): ~34 fps, imagem lavada |
| `lag2` ou menos | Ainda lavada |
| `lag3`, `lag4` | Estável, igual ao Eglantine (o padrão é `lag3`) |
| `lag6` ou mais | Pisca: o upload do mirror lê a memória do jogo quando a GPU executa, e o jogo já a reescreveu |
| `replay` | Espera só o replay gravar o submit |
| `rlagN` | Espera o replay gravar o submit e a GPU de N submits atrás |

A causa do brilho: o jogo soma a luminância da cena num buffer de 0x1040 bytes (compute `435a0138b9013040`). A CPU lê
a soma, zera o buffer e dá flush a cada frame. Com a espera total, o zero chega sempre antes da soma do frame seguinte,
e a CPU lê 29. No console a GPU está adiantada, e o valor fica em 44, o mesmo do `lag3`.

### Replay mais leve (`a89bbd3`)

- `counter_word` varria a fila de relatórios de occlusion query pendentes a cada draw condicional. Com `lag3` há
  centenas em voo, e isso era 8,6% do thread de replay. Agora é uma tabela hash.
- `nvn2_origin` pegava um mutex e consultava um `unordered_map` várias vezes por draw (3,2%). Ganhou um cache por
  thread.
- `rt_live` sem `OOT_LIVE` lê cada variável de ambiente uma vez só.

### Captura das escritas da CPU no submit

A cada `nvnQueueSubmitCommands`, ainda na thread do jogo, o runtime consulta o write-watch de todos os mirrors e
carimba as páginas escritas. Em seguida copia **naquele instante** o conteúdo delas, desde a captura anterior, para um
buffer de staging. Ficam de fora as páginas cujo carimbo mais novo é o write-back do próprio mirror.

O primeiro prólogo daquele submit copia esse staging para os mirrors. Daí em diante o sync do mirror só sobe regiões
que o write-watch não rastreia, e os `FlushMappedRange` não sobem mais nada. Com isso, cada submit vê as escritas da CPU
feitas antes dele, como no hardware, e não o que o jogo escreveu depois enquanto o replay ou a GPU ainda não tinham
chegado lá.

O upload caiu de 48 para 1,3 MiB por frame. `OOT_CAPTURE` é 1 por padrão e 0 volta ao caminho antigo; dá para trocar
ao vivo.

Junto veio o pool de snapshots do merge, que passou de 16 para 96 MiB (`OOT_SNAP_POOL_MIB`). Com merges de até 16 MiB
a floresta faz 89 MiB de snapshots por frame, e o pool antigo obrigava a criar 13 buffers novos por frame.

Limite conhecido: o acumulador da soma de cena (`0x1040` bytes, compute `435a0138b9013040`) às vezes não zera por
alguns frames (44, 87, 131…), porque o write-back de um submit ainda em voo põe a soma por cima do zero que a CPU
escreveu na memória do jogo. A imagem não muda (brilho 60-62, igual ao Eglantine). Uma correção completa seria um
merge que soma a diferença da GPU, em vez de sobrescrever, nos buffers que ela acumula.

## Ajustes com o jogo aberto

`OOT_LIVE=/caminho/live.txt` faz o runtime reler esse arquivo a cada 250 ms. Cada linha tem a forma `NOME=VALOR`.
Funciona para `OOT_SUBMIT_SYNC`, `OOT_EV`, `OOT_MERGE_MAX_MIB`, `OOT_MIRROR_STAGE`, `TOTK_MIRROR_WB_RESYNC`,
`TOTK_FLUSH_TRUST`, `OOT_CAPTURE` e `OOT_SNAP_POOL_MIB`. Cada mudança aparece no log como `live: NOME=VALOR`.

## Experimentos que ficaram desligados

| Variável | O que faz | Por que está desligada |
|---|---|---|
| `OOT_MIRROR_STAGE=1` | Copia os uploads do mirror na hora do submit, não quando a GPU executa | Tira o pisca de `lag6+`, mas um lag maior não deu fps a mais |
| `TOTK_FLUSH_TRUST=1` | Num `FlushMappedRange`, sobe só as páginas que o write-watch viu escritas | 55-60 fps, mas a exposição oscila (brilho 62-94) |

Por que `TOTK_FLUSH_TRUST` falha: dos 46,8 MiB que sobem por frame a cada flush, 39 MiB são páginas que a própria GPU
escreveu. Subir tudo esconde uma corrida. O write-back de um submit põe a soma inteira do mirror por cima do zero que
a CPU escreveu enquanto ele rodava, e a CPU passa a ler 29, 44, 74 ou 88 de um frame para outro. No console a GPU soma
direto na memória, então isso não acontece. O write-watch em si não perde nenhuma escrita: `OOT_FLUSH_AUDIT` faz o
hash das páginas flushadas e não encontrou nenhuma mudança sem registro.

## O que falta para 60 fps

- **O thread de replay.** A GPU trabalha ~13 ms por frame, mas fica ~5 ms parada esperando o próximo command buffer:
  o replay leva ~15 ms para gravar um frame. O perfil é espalhado: `bind_resources` 12%, cópias e comparações de
  memória ~11%, `vk_usnap_find` 3%, a fila do recorder 3%. Medido sem efeito:
  - partes menores de command buffer (`TOTK_SPLIT_US`, `TOTK_SPLIT_DRAWS`);
  - lag maior (`lag4`, `lag5`);
  - compilar os computes grandes com otimização (o driver já otimiza, os tempos de GPU não mudaram);
  - baixar a prioridade da thread do driver.
- **`lag6` ou mais ainda pisca**, mesmo com a captura. Sobra o snapshot de uniforms, que é tirado quando o replay grava,
  e não no submit.
- **Uma thread do driver NVIDIA** (aparece com o nome `guest main`, dentro de `_nv002nvvm`) fica a 100% de um núcleo
  desde o início do gameplay. O driver está compilando algo em segundo plano. Não é a rede de upscale do vídeo, e
  `__GL_THREADED_OPTIMIZATIONS=0` não muda nada. Baixar a prioridade dela também não mudou o fps.
- **A memória do processo** pula de ~9,3 para ~14 GB por volta dos 4 minutos e meio de jogo e depois fica estável. Isso
  acontece com e sem a captura, então não vem destas mudanças.
- **60 fps num monitor de 100 Hz tremem**, porque os frames alternam entre 1 e 2 atualizações da tela. Use o monitor
  em 60 ou 120 Hz.

## Diagnósticos (`LOCAL TEST`)

| Variável | Mostra |
|---|---|
| `OOT_GPU_PROF=<frame>` (com `TOTK_VK_RECORDER=0`) | Tempo de GPU por pass e por dispatch em 6 frames |
| `OOT_WB_LOG=<frame>` | Cada binding de storage escrito naquele frame, e as páginas que cada sync sobe |
| `OOT_INV_LOG=<frame>` | Leituras da CPU de memória escrita pela GPU (pool invalidate de 0x1040 bytes) por 40 frames |
| `OOT_FLUSH_AUDIT=<frame>` | Páginas flushadas que mudaram sem o write-watch ver |
| `OOT_EV_LOG=1` | O bloco de uniforms do tone map |
| `TOTK_TRACE=perf` | Linha do mirror: upload, write-back, merge, flush e as páginas da GPU dentro dos flushes |
