// nn::codec video decoding (the Switch 2 SDK's VP9 path) on the host with libvpx, for Donkey Kong Bananza's tutorial
// guide movies (Movie/Guide*.webm, 960x540). The SDK's nn::codec::video::Initialize opens the console's video service
// over IPC, which the host does not serve (the first guide movie crashed in nn::codec::video::detail::ipc::Initialize),
// so the decoder calls are host functions. The demuxer (nn::movie::MediaExtractor) stays guest code.
// What the game's movie player (main+0xdfb8xx..0xdfe9xx) expects, read from its code:
// - each output buffer i is an image pool of its own, tied to i by RegisterVp9DecoderBuffer; the player reads the
//   pool's size at +0x20 / +0x22 (u16 width, height), which InitializeDecoderImagePool takes from the preference
//   (MakeDecodeImagePreference, SDK code: +8 u16 width - 1, +10 u16 height - 1);
// - it acquires an output buffer before it submits input (its wrapper aborts on any failure there): Acquire reserves
//   the buffer the next shown picture goes to and returns a fence (NativeSyncHandle, which the caller zeroed; the
//   host writes a token in its first word); the player waits for it through nn::syncpt::sdk::RequestSyncPointEvent,
//   signalled when SubmitVp9DecoderInputBuffer has decoded a picture into that buffer (at once if it already has);
// - a submission with no reserved buffer to decode into answers "not ready" (the wrapper's retry code 0x76);
// - the picture is read with ReadDecoderOutputImageNv12 (Y plane, then interleaved UV, each with the caller's pitch).
//   GetDecoderImagePoolNativeImageHandle (the player's GPU path) has no image yet and returns 0.
// libvpx is BSD-3-Clause and already linked for ov_movie.c.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <vpx/vp8dx.h>
#include <errno.h>
#include <dav1d/dav1d.h>
#include <vpx/vpx_decoder.h>

#include "host.h"
#include "platform/platform.h"

enum { R_NOT_READY = 0x1c266f };   // (result & 0x3fffff) - 0x1c2000 == 0x66f: "try again" for the game's wrapper
// QUEUED: packets the decoder takes ahead of the buffers for their pictures, as Eglantine's video session does
// (nx2_gsv.h, INx2VideoSession::MaxQueuedInputs); a submission past them answers "not ready".
enum { SLOTS = 16, DECODERS = 4, WAITERS = 4, QUEUED = 8 };
enum { S_FREE, S_WAIT, S_FILLED };   // registered; acquired with no picture yet; acquired holding a picture
enum { FENCE_TAG = 0xc0de0000u };    // fence token: tag | decoder << 8 | buffer

// TOTK_TRACE=codec: a line for each decoder call that moves a buffer or a picture, to follow a player that stops.
#define TRACE(...)                                                                                                   \
    do {                                                                                                             \
        if (rt_trace & RT_TRACE_CODEC) rt_log("codec: " __VA_ARGS__);                                                \
    } while (0)

#define LOG_FIRST(...)                                                                                               \
    do {                                                                                                             \
        static volatile int32_t logged;                                                                              \
        if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED)) rt_log(__VA_ARGS__);                                 \
    } while (0)

typedef struct {
    uint64_t pool;     // guest VideoDecoderImagePoolType registered for this index; 0: not registered
    int state;
    int fence_done;    // the buffer's fence has fired: it stays so until the buffer is acquired again, even if a flush drops
                       // the picture (movie::VideoDecoder's callback aborts when a fired fence reads as not signalled)
    uint32_t seq;      // acquisition order: pictures fill the oldest waiting buffer
    uint64_t waiter[WAITERS];   // guest system events of sync point events requested before the picture came
    uint16_t w, h;
    int64_t pts;       // nanoseconds (nn::TimeSpan)
    int eos;
    uint8_t *nv12;     // w*h luma, then cw*ch interleaved UV pairs
} Slot;

// A packet submitted while no buffer was reserved for its picture, decoded once one is (ov_vp9_acquire).
typedef struct { uint8_t *data; size_t size; int64_t pts; int end; } Packet;

typedef struct {
    uint64_t key;      // guest Vp9DecoderType; 0: unused
    vpx_codec_ctx_t vpx;
    int vpx_ok;
    Dav1dContext *av1;   // set: an AV1 decoder (dav1d), not libvpx
    uint32_t seq;
    Slot slot[SLOTS];
    Slot last;         // the newest picture, for packets that show none
    Packet queue[QUEUED];
    int qhead, qn;
    uint32_t gen;      // bumped by init, flush and finalize: a packet the worker took before it is dropped
    int pending;       // packets handed to the worker whose pictures will fill waiting buffers
} Dec;

static Dec decs[DECODERS];
static PlatMutex lock = PLAT_MUTEX_INIT;   // every decoder call; held across a decode (one 960x540 frame, a few ms)

static Dec *dec_of(uint64_t key) {
    for (int i = 0; i < DECODERS; i++)
        if (decs[i].key == key) return &decs[i];
    return NULL;
}

static uint32_t cw(uint16_t w) { return (w + 1u) / 2; }
static uint32_t ch(uint16_t h) { return (h + 1u) / 2; }

// Copies a decoded I420 picture into s as NV12.
static void put_picture(Slot *s, const vpx_image_t *img) {
    uint16_t w = (uint16_t)img->d_w, h = (uint16_t)img->d_h;
    if (!s->nv12 || s->w != w || s->h != h) {
        free(s->nv12);
        s->nv12 = malloc((size_t)w * h + 2 * (size_t)cw(w) * ch(h));
        if (!s->nv12) rt_fatal("codec: out of memory for a %ux%u picture", w, h);
        s->w = w, s->h = h;
    }
    for (uint32_t y = 0; y < h; y++) memcpy(s->nv12 + (size_t)y * w, img->planes[0] + (size_t)y * img->stride[0], w);
    uint8_t *uv = s->nv12 + (size_t)w * h;
    for (uint32_t y = 0; y < ch(h); y++) {
        const uint8_t *u = img->planes[1] + (size_t)y * img->stride[1], *v = img->planes[2] + (size_t)y * img->stride[2];
        uint8_t *o = uv + (size_t)y * 2 * cw(w);
        for (uint32_t x = 0; x < cw(w); x++) o[2 * x] = u[x], o[2 * x + 1] = v[x];
    }
}

static void copy_slot(Slot *to, const Slot *from) {
    if (!from->nv12) return;
    size_t n = (size_t)from->w * from->h + 2 * (size_t)cw(from->w) * ch(from->h);
    if (!to->nv12 || to->w != from->w || to->h != from->h) {
        free(to->nv12);
        if (!(to->nv12 = malloc(n))) rt_fatal("codec: out of memory for a picture");
        to->w = from->w, to->h = from->h;
    }
    memcpy(to->nv12, from->nv12, n);
}

static void queue_drop(Dec *d) {
    for (int i = 0; i < d->qn; i++) free(d->queue[(d->qhead + i) % QUEUED].data);
    d->qhead = d->qn = 0;
}

// The oldest reserved buffer still waiting for a picture: pictures fill buffers in acquisition order.
static Slot *oldest_wait(Dec *d) {
    Slot *s = NULL;
    for (int i = 0; i < SLOTS; i++)
        if (d->slot[i].state == S_WAIT && (!s || (int32_t)(d->slot[i].seq - s->seq) < 0)) s = &d->slot[i];
    return s;
}

// Decodes one packet into the waiting buffer s (lock held). A packet that shows nothing leaves s waiting, except the
// last one: the player then gets the previous picture (or black) marked as the end. Returns the number of events to
// signal, stored in wake.
// One AV1 packet through dav1d. Returns 1 with the newest picture as an image (8-bit; 10-bit is shifted down), which stays
// valid until the next call. ponytail: HDR is not tone mapped, and only 4:2:0 is expected.
static void av1_no_free(const uint8_t *buf, void *cookie) { (void)buf; (void)cookie; }
static int av1_decode(Dec *d, const uint8_t *data, size_t size, vpx_image_t *out) {
    static Dav1dPicture pic;
    static int have;
    static uint8_t *conv[3];
    static size_t conv_cap[3];
    Dav1dData in = {0};
    if (dav1d_data_wrap(&in, data, size, av1_no_free, NULL) < 0) return 0;
    int got = 0;
    while (in.sz) {
        int r = dav1d_send_data(d->av1, &in);
        if (r < 0 && r != DAV1D_ERR(EAGAIN)) {
            LOG_FIRST("codec: AV1 packet of %zu bytes failed: %d", size, r);
            dav1d_data_unref(&in);
            break;
        }
        for (;;) {
            Dav1dPicture next = {0};
            r = dav1d_get_picture(d->av1, &next);
            if (r < 0) break;
            if (have) dav1d_picture_unref(&pic);
            pic = next, have = 1, got = 1;
        }
    }
    if (!got) return 0;
    *out = (vpx_image_t){.d_w = (unsigned)pic.p.w, .d_h = (unsigned)pic.p.h};
    if (pic.p.bpc == 8) {
        for (int i = 0; i < 3; i++) out->planes[i] = pic.data[i], out->stride[i] = (int)pic.stride[i ? 1 : 0];
        return 1;
    }
    int shift = pic.p.bpc - 8;
    for (int i = 0; i < 3; i++) {
        unsigned w = i ? (pic.p.w + 1u) / 2 : (unsigned)pic.p.w, h = i ? (pic.p.h + 1u) / 2 : (unsigned)pic.p.h;
        if (conv_cap[i] < (size_t)w * h) {
            free(conv[i]);
            conv[i] = malloc((size_t)w * h);
            if (!conv[i]) rt_fatal("codec: out of memory for an AV1 picture");
            conv_cap[i] = (size_t)w * h;
        }
        for (unsigned y = 0; y < h; y++) {
            const uint16_t *in16 = (const uint16_t *)((const uint8_t *)pic.data[i] + (ptrdiff_t)y * pic.stride[i ? 1 : 0]);
            for (unsigned x = 0; x < w; x++) conv[i][(size_t)y * w + x] = (uint8_t)(in16[x] >> shift);
        }
        out->planes[i] = conv[i], out->stride[i] = (int)w;
    }
    return 1;
}

// TOTK_TRACE=perf: wall time of the decode and of the NV12 conversion, every 300 pictures.
static struct { uint64_t n, dec_ns, conv_ns, max_ns; } dstat;

// The decoder's part of a packet: the picture it shows, or NULL. The image stays valid until the next call. Called by
// one thread at a time: the codec worker, or a guest thread holding `lock` when decoding is synchronous.
static const vpx_image_t *decode_packet(Dec *d, const uint8_t *data, size_t size, vpx_image_t *av1_img) {
    const vpx_image_t *img = NULL, *f;
    if (d->av1) {
        if (data && size && av1_decode(d, data, size, av1_img)) img = av1_img;
    } else if (d->vpx_ok && data && size) {
        if (vpx_codec_decode(&d->vpx, data, (unsigned)size, NULL, 0) != VPX_CODEC_OK)
            LOG_FIRST("codec: VP9 packet of %zu bytes failed: %s", size, vpx_codec_error_detail(&d->vpx));
        vpx_codec_iter_t it = NULL;
        while ((f = vpx_codec_get_frame(&d->vpx, &it))) img = f;   // a superframe shows at most one; keep the last
    }
    return img;
}

// The buffer's part (lock held): s gets img, or at the end the last picture (or black). t0..t1: the decode's wall time.
static int finish_into(Dec *d, Slot *s, const vpx_image_t *img, size_t size, int64_t pts, int end, uint64_t wake[WAITERS],
                       uint64_t t0, uint64_t t1) {
    int filled = 1;
    if (img) put_picture(s, img), copy_slot(&d->last, s);
    if (img && (rt_trace & RT_TRACE_PERF)) {
        uint64_t t2 = plat_now_ns();
        dstat.n++, dstat.dec_ns += t1 - t0, dstat.conv_ns += t2 - t1;
        if (t2 - t0 > dstat.max_ns) dstat.max_ns = t2 - t0;
        if (dstat.n == 300) {
            rt_log("perf: codec: %llu pictures, decode %.2f ms avg, conversion and copies %.2f ms avg, slowest %.2f ms",
                   (unsigned long long)dstat.n, dstat.dec_ns / 1e6 / dstat.n, dstat.conv_ns / 1e6 / dstat.n, dstat.max_ns / 1e6);
            memset(&dstat, 0, sizeof dstat);
        }
    }
    if (img) {   // LOCAL TEST: is the movie really decoded? The mean luma of the first pictures
        static int shown;
        if (shown++ < 8 || shown % 300 == 0) {
            uint64_t sum = 0;
            for (uint32_t y = 0; y < s->h; y += 16)
                for (uint32_t x = 0; x < s->w; x += 16) sum += s->nv12[(size_t)y * s->w + x];
            rt_log("codec: picture %d %ux%u mean luma %llu", shown, s->w, s->h, (unsigned long long)(sum / (((s->h + 15) / 16) * ((s->w + 15) / 16))));
        }
    }
    else if (end && d->last.nv12) copy_slot(s, &d->last);
    else if (end) {
        vpx_image_t black = {.d_w = 16, .d_h = 16};
        static uint8_t y[16 * 16], uv[8 * 8];
        if (!uv[0]) memset(y, 16, sizeof y), memset(uv, 128, sizeof uv);
        black.planes[0] = y, black.planes[1] = black.planes[2] = uv, black.stride[0] = 16, black.stride[1] = black.stride[2] = 8;
        put_picture(s, &black);
    } else filled = 0;
    TRACE("decode %zu bytes pts %lld%s: buffer %d %s", size, (long long)pts, end ? " end" : "", (int)(s - d->slot),
          filled ? "filled" : "still waiting (no picture shown)");
    LOG_FIRST("codec: first VP9 packet: %zu bytes, picture %ux%u", size, s->w, s->h);
    int n = 0;
    if (filled) {
        s->state = S_FILLED, s->fence_done = 1, s->pts = pts, s->eos = end;
        for (int k = 0; k < WAITERS; k++)
            if (s->waiter[k]) wake[n++] = s->waiter[k];
        memset(s->waiter, 0, sizeof s->waiter);
    }
    return n;
}

// Decodes one packet into the waiting buffer s on the calling thread (lock held): the synchronous path.
static int decode_into(Dec *d, Slot *s, const uint8_t *data, size_t size, int64_t pts, int end, uint64_t wake[WAITERS]) {
    vpx_image_t av1_img;
    uint64_t t0 = plat_now_ns();
    const vpx_image_t *img = decode_packet(d, data, size, &av1_img);
    return finish_into(d, s, img, size, pts, end, wake, t0, plat_now_ns());
}

// ---- asynchronous decoding (OOT_CODEC_SYNC=1 decodes on the guest thread inside Submit and Acquire, as before)
// The player waits for a picture through its buffer's fence (RequestSyncPointEvent, TryWaitNativeSync), so Submit need
// not return with the picture decoded. A 1080p AV1 picture took 12-15 ms of the guest thread on a Ryzen 5 5600 (the
// title screen's movie), most of a frame. A host thread takes the packets in submission order, decodes them without
// `lock`, then fills the oldest waiting buffer and signals its waiters: the buffer the guest-thread decode would have
// chosen, as pictures fill buffers in acquisition order either way.
enum { JOBS = DECODERS * (SLOTS + QUEUED) };
typedef struct { Dec *d; uint32_t gen; Packet p; } Job;
static Job jobs[JOBS];
static int job_head, job_n;
static Dec *busy;   // the decoder the worker is decoding for, outside `lock`
static PlatCond jcond = PLAT_COND_INIT, idle = PLAT_COND_INIT;
static int worker;  // 0 not started, 1 running, -1 could not start (synchronous from then on)

static int codec_async(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("OOT_CODEC_SYNC");
        on = !(e && *e && *e != '0');
    }
    return on && worker >= 0;
}

// Waiting buffers that no packet handed to the worker is headed for (lock held).
static int avail(const Dec *d) {
    int n = 0;
    for (int i = 0; i < SLOTS; i++) n += d->slot[i].state == S_WAIT;
    return n - d->pending;
}

static void job_push(Dec *d, Packet p) {   // lock held, job_n < JOBS
    jobs[(job_head + job_n++) % JOBS] = (Job){d, d->gen, p};
    d->pending++;
    plat_cond_signal(&jcond);
}

// Hands queued packets to the worker while buffers wait for their pictures (lock held).
static void feed(Dec *d) {
    while (d->qn && avail(d) > 0 && job_n < JOBS) {
        Packet q = d->queue[d->qhead];
        d->qhead = (d->qhead + 1) % QUEUED, d->qn--;
        job_push(d, q);
    }
}

// Drops d's packets not decoded yet and waits until the worker is not decoding for d, so the caller may close, flush
// or reset the decoder (lock held).
static void quiesce(Dec *d) {
    d->gen++;
    int k = 0;
    for (int i = 0; i < job_n; i++) {
        Job j = jobs[(job_head + i) % JOBS];
        if (j.d == d) free(j.p.data);
        else jobs[(job_head + k++) % JOBS] = j;
    }
    job_n = k;
    while (busy == d) plat_cond_wait(&idle, &lock);
    d->pending = 0;
}

static void worker_main(void *arg) {
    (void)arg;
    plat_mutex_lock(&lock);
    for (;;) {
        while (!job_n) plat_cond_wait(&jcond, &lock);
        Job j = jobs[job_head];
        job_head = (job_head + 1) % JOBS, job_n--;
        Dec *d = j.d;
        if (j.gen != d->gen) {
            free(j.p.data);
            continue;
        }
        busy = d;
        plat_mutex_unlock(&lock);
        vpx_image_t av1_img;
        uint64_t t0 = plat_now_ns();
        const vpx_image_t *img = decode_packet(d, j.p.data, j.p.size, &av1_img);
        uint64_t t1 = plat_now_ns();
        plat_mutex_lock(&lock);
        busy = NULL;
        uint64_t wake[WAITERS];
        int n = 0;
        if (j.gen == d->gen) {
            d->pending--;
            Slot *s = oldest_wait(d);
            if (s) n = finish_into(d, s, img, j.p.size, j.p.pts, j.p.end, wake, t0, t1);
            feed(d);
        }
        plat_cond_broadcast(&idle);
        free(j.p.data);
        if (n) {
            plat_mutex_unlock(&lock);
            for (int i = 0; i < n; i++) kern_signal_system_event(wake[i]);
            plat_mutex_lock(&lock);
        }
    }
}

// Result nn::codec::video::Initialize(), Finalize(): nothing to open on the host
static void ov_video_init(Cpu *c) { ret_x(c, 0); }
static void ov_video_fini(Cpu *c) { ret_x(c, 0); }

static void vp9_init(Cpu *c, uint64_t key, unsigned w, unsigned h) {
    plat_mutex_lock(&lock);
    Dec *d = dec_of(key);
    if (!d) d = dec_of(0);
    if (!d) rt_fatal("codec: more than %d VP9 decoders", DECODERS);
    quiesce(d);
    uint32_t gen = d->gen;
    if (d->vpx_ok) vpx_codec_destroy(&d->vpx);
    if (d->av1) dav1d_close(&d->av1);
    for (int i = 0; i < SLOTS; i++) free(d->slot[i].nv12);
    free(d->last.nv12);
    queue_drop(d);
    memset(d, 0, sizeof *d);
    d->key = key, d->gen = gen;
    vpx_codec_dec_cfg_t cfg = {.threads = 2, .w = w, .h = h};
    d->vpx_ok = vpx_codec_dec_init(&d->vpx, vpx_codec_vp9_dx(), &cfg, 0) == VPX_CODEC_OK;
    plat_mutex_unlock(&lock);
    rt_log("codec: VP9 decoder %#llx %ux%u on the host (libvpx%s)", (unsigned long long)key, w, h,
           d->vpx_ok ? "" : " failed to start: pictures stay black");
    ret_x(c, 0);
}
// Result InitializeVp9Decoder(Vp9DecoderType *, void *work, size_t, size_t, u16 w, u16 h, u32, input, output fmt)
static void ov_vp9_init(Cpu *c) { vp9_init(c, arg_x(c, 0), arg_x(c, 4) & 0xffff, arg_x(c, 5) & 0xffff); }
// Result InitializeVp9Decoder(Vp9DecoderType *, size_t, size_t, u16 w, u16 h, u32, input, output fmt)
static void ov_vp9_init_nowork(Cpu *c) { vp9_init(c, arg_x(c, 0), arg_x(c, 3) & 0xffff, arg_x(c, 4) & 0xffff); }

// Result FinalizeVp9Decoder(Vp9DecoderType *)
static void ov_vp9_fini(Cpu *c) {
    plat_mutex_lock(&lock);
    Dec *d = dec_of(arg_x(c, 0));
    if (d) {
        quiesce(d);
        uint32_t gen = d->gen;
        if (d->vpx_ok) vpx_codec_destroy(&d->vpx);
        if (d->av1) dav1d_close(&d->av1);
        for (int i = 0; i < SLOTS; i++) free(d->slot[i].nv12);
        free(d->last.nv12);
        queue_drop(d);
        memset(d, 0, sizeof *d);
        d->gen = gen;
    }
    plat_mutex_unlock(&lock);
    ret_x(c, 0);
}

// Result FlushVp9Decoder(Vp9DecoderType *): outputs not handed over yet are dropped
static void ov_vp9_flush(Cpu *c) {
    plat_mutex_lock(&lock);
    Dec *d = dec_of(arg_x(c, 0));
    if (d) quiesce(d);   // packets handed to the worker are dropped too
    if (d)
        for (int i = 0; i < SLOTS; i++)
            if (d->slot[i].state == S_FILLED) d->slot[i].state = S_WAIT;   // still reserved; its picture is dropped
    if (d) queue_drop(d);   // and so are the packets not decoded yet
    plat_mutex_unlock(&lock);
    TRACE("flush");
    ret_x(c, 0);
}

// Result Start/StopVp9Decoder(Vp9DecoderType *), FinalizeDecoderImagePool(pool): nothing to do
static void ov_ok(Cpu *c) { ret_x(c, 0); }

// Result RegisterVp9DecoderBuffer(Vp9DecoderType *, u32 index, VideoDecoderImagePoolType *, u32)
static void ov_vp9_register(Cpu *c) {
    uint32_t i = (uint32_t)arg_x(c, 1);
    plat_mutex_lock(&lock);
    Dec *d = dec_of(arg_x(c, 0));
    if (d && i < SLOTS) d->slot[i].pool = arg_x(c, 2), d->slot[i].state = S_FREE;
    plat_mutex_unlock(&lock);
    if (!d || i >= SLOTS) rt_log("codec: RegisterVp9DecoderBuffer index %u of an unknown decoder or past %d", i, SLOTS);
    ret_x(c, 0);
}

// Result InitializeDecoderImagePool(pool, void *, size_t, const VideoDecoderImagePreference &, u32, CodecMode): the
// size the player reads back at +0x20 / +0x22
static void ov_pool_init(Cpu *c) {
    uint8_t *pool = arg_p(c, 0);
    const uint8_t *pref = arg_p(c, 3);
    uint16_t w, h;
    memcpy(&w, pref + 8, 2), memcpy(&h, pref + 10, 2);
    w++, h++;
    memcpy(pool + 0x20, &w, 2), memcpy(pool + 0x22, &h, 2);
    ret_x(c, 0);
}

// Result CalculateDecoderImagePoolBuffer(size_t *size, size_t *align, const preference &, u32 count),
// CalculateVp9DecoderWorkBuffer(size_t *size, size_t *align, size_t, u16 w, u16 h): the buffers hold nothing on the
// host (the pictures are host memory), so one page each
static void ov_calc_size(Cpu *c) {
    *(uint64_t *)arg_p(c, 0) = 0x1000;
    *(uint64_t *)arg_p(c, 1) = 0x1000;
    ret_x(c, 0);
}

// A guest SystemEventType (manual clear) standing for the game object at `key`: the same one for the same object, so
// movie after movie reuses the events instead of filling the kernel's table of bound events. Guest thread only.
// ponytail: 64 objects, linear scan; the player has one decoder and 15 sync point events.
static uint64_t event_for(Cpu *c, uint64_t key) {
    static struct { uint64_t key, ev; } tab[64];
    static int n;
    static PlatMutex tlock = PLAT_MUTEX_INIT;
    plat_mutex_lock(&tlock);
    uint64_t ev = 0;
    for (int i = 0; i < n && !ev; i++)
        if (tab[i].key == key) ev = tab[i].ev;
    if (!ev) {
        if (n == 64) rt_fatal("codec: more than 64 decoder and sync point objects");
        ev = (uint64_t)(uintptr_t)mem_kernel_alloc(0x100, 16);
        tab[n].key = key, tab[n++].ev = ev;
    }
    plat_mutex_unlock(&tlock);
    kern_bind_system_event(c, ev, 0);   // once per event; leaves the event's handles in x0-x2
    return ev;
}

// Result GetVp9DecoderSubmissionReadyEvent(nn::os::SystemEventType **out, Vp9DecoderType *): signalled while the
// decoder can take a packet, which is when an output buffer is reserved for its picture. The player clears it after a
// submission comes back not ready, so AcquireVp9DecoderOutputBuffer signals it again (ov_vp9_acquire).
static void ov_vp9_ready_event(Cpu *c) {
    uint64_t *out = arg_p(c, 0), dec = arg_x(c, 1);
    uint64_t ev = event_for(c, dec);
    kern_signal_system_event(ev);
    *out = ev;
    TRACE("submission-ready event %#llx handed out (signalled)", (unsigned long long)ev);
    ret_x(c, 0);
}

// Result SubmitVp9DecoderInputBuffer(u64 *id, Vp9DecoderType *, const void *data, size_t size, TimeSpan pts, bool end):
// decoded at once into the oldest waiting buffer, else queued for the next one reserved; "not ready" once QUEUED
// packets wait. After the end of a guide movie the player flushes and starts over with the buffers it still holds,
// submitting its third packet before it releases one: without the queue that packet was refused, nothing released a
// buffer again, and the window kept the movie's first picture, a flat yellow card.
static void ov_vp9_submit(Cpu *c) {
    uint64_t *id = arg_p(c, 0);
    const uint8_t *data = arg_p(c, 2);
    size_t size = arg_x(c, 3);
    int64_t pts = (int64_t)arg_x(c, 4);
    int end = (int)(arg_x(c, 5) & 1);
    plat_mutex_lock(&lock);
    Dec *d = dec_of(arg_x(c, 1));
    if (d && codec_async() && !worker) worker = plat_thread_spawn(worker_main, NULL, "codec", 0) ? 1 : -1;
    if (d && codec_async()) {   // the packet goes to the worker, or waits in the queue for a buffer
        int direct = !d->qn && avail(d) > 0 && job_n < JOBS, queued = !direct && d->qn < QUEUED;
        if (direct || queued) {
            Packet q = {size ? malloc(size) : NULL, size, pts, end};
            if (size && !q.data) rt_fatal("codec: out of memory for a %zu-byte packet", size);
            if (size) memcpy(q.data, data, size);
            if (direct) job_push(d, q);
            else d->queue[(d->qhead + d->qn++) % QUEUED] = q;
            if (id) *id = d->seq;
        }
        int waiting = d->qn;
        plat_mutex_unlock(&lock);
        TRACE("submit %zu bytes pts %lld%s: %s (%d queued)", size, (long long)pts, end ? " end" : "",
              direct ? "to the decoder thread" : queued ? "queued for the next reserved buffer" : "not ready", waiting);
        ret_x(c, direct || queued ? 0 : R_NOT_READY);
        return;
    }
    Slot *s = d && !d->qn ? oldest_wait(d) : NULL;
    if (!s) {
        int queued = d && d->qn < QUEUED;
        if (queued) {
            Packet *q = &d->queue[(d->qhead + d->qn++) % QUEUED];
            *q = (Packet){size ? malloc(size) : NULL, size, pts, end};
            if (size && !q->data) rt_fatal("codec: out of memory for a %zu-byte packet", size);
            if (size) memcpy(q->data, data, size);
            if (id) *id = d->seq;
        }
        int waiting = d ? d->qn : 0;
        plat_mutex_unlock(&lock);
        TRACE("submit %zu bytes pts %lld%s: %s (%d queued)", size, (long long)pts, end ? " end" : "",
              queued ? "queued for the next reserved buffer" : "not ready", waiting);
        ret_x(c, queued ? 0 : R_NOT_READY);
        return;
    }
    uint64_t wake[WAITERS];
    int n = decode_into(d, s, data, size, pts, end, wake);
    if (id) *id = d->seq;
    plat_mutex_unlock(&lock);
    for (int k = 0; k < n; k++) kern_signal_system_event(wake[k]);
    ret_x(c, 0);
}

// Result AcquireVp9DecoderOutputBuffer(u32 *index, NativeSyncHandle *fence, Vp9DecoderType *): reserves a free buffer
// for the next picture, decodes queued packets into the waiting buffers; the fence carries the token
// RequestSyncPointEvent waits on
static void ov_vp9_acquire(Cpu *c) {
    uint32_t *index = arg_p(c, 0), *fence = arg_p(c, 1);
    uint64_t key = arg_x(c, 2);
    uint64_t wake[SLOTS * WAITERS];
    int n = 0;
    plat_mutex_lock(&lock);
    Dec *d = dec_of(key);
    int got = -1;
    for (int i = 0; d && i < SLOTS && got < 0; i++)
        if (d->slot[i].pool && d->slot[i].state == S_FREE) got = i;
    if (got >= 0) {
        Slot *s = &d->slot[got];
        s->state = S_WAIT, s->fence_done = 0, s->seq = d->seq++, s->eos = 0;
        memset(s->waiter, 0, sizeof s->waiter);
        *index = (uint32_t)got;
        if (fence) *fence = FENCE_TAG | (uint32_t)(d - decs) << 8 | (uint32_t)got;
        Slot *w;
        if (codec_async() && worker > 0) feed(d);   // the worker decodes them and signals the fences
        else while (d->qn && (w = oldest_wait(d)) && n <= SLOTS * WAITERS - WAITERS) {
            Packet q = d->queue[d->qhead];
            d->qhead = (d->qhead + 1) % QUEUED, d->qn--;
            n += decode_into(d, w, q.data, q.size, q.pts, q.end, wake + n);
            free(q.data);
        }
    }
    plat_mutex_unlock(&lock);
    if (got < 0) rt_log("codec: AcquireVp9DecoderOutputBuffer with every registered buffer reserved");
    else TRACE("acquire: buffer %d", got);
    for (int k = 0; k < n; k++) kern_signal_system_event(wake[k]);
    // A reserved buffer lets the next packet in: wake a player that waits for the submission-ready event after a not
    // ready submission. Without this, the tutorial's guide movies stopped at their second picture.
    if (got >= 0) kern_signal_system_event(event_for(c, key));
    ret_x(c, got < 0 ? R_NOT_READY : 0);
}

// Result ReleaseVp9DecoderOutputBuffer(Vp9DecoderType *, u32 index, NativeSyncHandle)
static void ov_vp9_release(Cpu *c) {
    uint32_t i = (uint32_t)arg_x(c, 1);
    plat_mutex_lock(&lock);
    Dec *d = dec_of(arg_x(c, 0));
    if (d && i < SLOTS && d->slot[i].state != S_FREE) d->slot[i].state = S_FREE;
    plat_mutex_unlock(&lock);
    TRACE("release: buffer %u", i);
    ret_x(c, 0);
}

// Result GetVp9DecoderOutputBufferProperties(u16 *w, u16 *h, u16 *, u16 *, nn::movie::ColorSpace *, TimeSpan *pts,
// bool *end, Vp9DecoderType *, u32 index). The two unnamed u16 get the picture size too (aligned or display size).
static void ov_vp9_props(Cpu *c) {
    uint32_t i = (uint32_t)arg_x(c, 8);
    plat_mutex_lock(&lock);
    Dec *d = dec_of(arg_x(c, 7));
    Slot s = d && i < SLOTS ? d->slot[i] : (Slot){0};
    plat_mutex_unlock(&lock);
    for (int k = 0; k < 4; k++) {
        uint16_t *p = arg_p(c, k);
        if (p) *p = k & 1 ? s.h : s.w;
    }
    LOG_FIRST("codec: output properties: %ux%u", s.w, s.h);
    if (arg_p(c, 4)) *(uint32_t *)arg_p(c, 4) = 0;
    if (arg_p(c, 5)) *(int64_t *)arg_p(c, 5) = s.pts;
    if (arg_p(c, 6)) *(uint8_t *)arg_p(c, 6) = (uint8_t)s.eos;
    ret_x(c, d ? 0 : R_NOT_READY);
}

// Result ReadDecoderOutputImageNv12(const VideoDecoderImagePoolType *, u32 image, u16 x, u16 y, u16 w, u16 h,
// void *luma, size_t luma_pitch, size_t luma_size, void *chroma, size_t chroma_pitch, size_t chroma_size)
static void ov_read_nv12(Cpu *c) {
    uint64_t pool = arg_x(c, 0);
    uint32_t x0 = arg_x(c, 2) & 0xffff, y0 = arg_x(c, 3) & 0xffff, w = arg_x(c, 4) & 0xffff, h = arg_x(c, 5) & 0xffff;
    uint8_t *ly = arg_p(c, 6), *luv = arg_p(c, 9);
    size_t yp = arg_x(c, 7), ys = arg_x(c, 8), uvp = arg_x(c, 10), uvs = arg_x(c, 11);
    plat_mutex_lock(&lock);
    const Slot *s = NULL;
    for (int k = 0; k < DECODERS && !s; k++)
        for (int i = 0; decs[k].key && i < SLOTS && !s; i++)
            if (decs[k].slot[i].pool == pool && decs[k].slot[i].nv12) s = &decs[k].slot[i];
    if (s) {
        uint32_t cx = x0 < s->w ? x0 : s->w, cy = y0 < s->h ? y0 : s->h;
        uint32_t rw = w < s->w - cx ? w : s->w - cx, rh = h < s->h - cy ? h : s->h - cy;
        size_t n = rw < yp ? rw : yp;   // each row stays inside the caller's pitch and buffer
        for (uint32_t r = 0; r < rh && ly && r * yp + n <= ys; r++)
            memcpy(ly + r * yp, s->nv12 + (size_t)(cy + r) * s->w + cx, n);
        const uint8_t *uv = s->nv12 + (size_t)s->w * s->h;
        uint32_t uw = 2 * ((rw + 1) / 2), uh = (rh + 1) / 2, ux = 2 * (cx / 2), uy = cy / 2;
        n = uw < uvp ? uw : uvp;
        for (uint32_t r = 0; r < uh && luv && r * uvp + n <= uvs; r++)
            memcpy(luv + r * uvp, uv + (size_t)(uy + r) * 2 * cw(s->w) + ux, n);
    }
    int64_t pts = s ? s->pts : -1;
    plat_mutex_unlock(&lock);
    LOG_FIRST("codec: first NV12 read %ux%u (%s)", w, h, s ? "a decoded picture" : "no picture for this pool");
    TRACE("read pool %#llx: pts %lld", (unsigned long long)pool, (long long)pts);
    ret_x(c, 0);
}

// NativeImageHandle GetDecoderImagePoolNativeImageHandle(const VideoDecoderImagePoolType *, u32): no GPU image
static void ov_pool_image(Cpu *c) {
    LOG_FIRST("codec: GetDecoderImagePoolNativeImageHandle %p #%u: no GPU image (the player's NV12 path is served)",
              arg_p(c, 0), (unsigned)arg_x(c, 1));
    ret_x(c, 0);
}

// ---- nn::syncpt::sdk: GPU sync point events (the decoder's output fences), not in the SDK's loaded modules. Each is a
// guest system event; a request is signalled at once, as the host has no sync point to wait for.
// Result CreateSyncPointEvent(SyncPointEventType *)
static void ov_sp_create(Cpu *c) {
    event_for(c, arg_x(c, 0));
    ret_x(c, 0);
}
// void InitializeMultiWaitHolder(nn::os::MultiWaitHolderType *, SyncPointEventType *): the system event's holder
static void ov_sp_holder(Cpu *c) {
    static uint64_t init;
    if (!init && !(init = ldr_symbol("_ZN2nn2os25InitializeMultiWaitHolderEPNS0_19MultiWaitHolderTypeEPNS0_15SystemEventTypeE")))
        rt_fatal("codec: nn::os::InitializeMultiWaitHolder(SystemEventType *) is not loaded");
    uint64_t holder = arg_x(c, 0), ev = event_for(c, arg_x(c, 1));
    rt_call(c, init, 2, holder, ev);
}
// bool TryWaitSyncPointEvent(const SyncPointEventType *)
static void ov_sp_trywait(Cpu *c) {
    static uint64_t trywait;
    if (!trywait && !(trywait = ldr_symbol("_ZN2nn2os18TryWaitSystemEventEPNS0_15SystemEventTypeE")))
        rt_fatal("codec: nn::os::TryWaitSystemEvent is not loaded");
    uint64_t ev = event_for(c, arg_x(c, 0));
    ret_x(c, rt_call(c, trywait, 1, ev) & 1);
}
static void clear_event(Cpu *c, uint64_t ev) {
    static uint64_t clear;
    if (!clear && !(clear = ldr_symbol("_ZN2nn2os16ClearSystemEventEPNS0_15SystemEventTypeE")))
        rt_fatal("codec: nn::os::ClearSystemEvent is not loaded");
    rt_call(c, clear, 1, ev);
}

// void RequestSyncPointEvent(SyncPointEventType *, const SyncPointHandle &): re-arms the event (cleared, as the last
// request may have left it signalled); a decoder fence (FENCE_TAG) fires when its buffer holds a picture, anything
// else at once, as the host has no sync point to wait for
static void ov_sp_request(Cpu *c) {
    const uint32_t *h = arg_p(c, 1);
    uint32_t tok = h ? *h : 0;
    uint64_t ev = event_for(c, arg_x(c, 0));
    clear_event(c, ev);
    int now = 1;
    if ((tok & 0xffff0000u) == FENCE_TAG && (tok >> 8 & 0xff) < DECODERS && (tok & 0xff) < SLOTS) {
        plat_mutex_lock(&lock);
        Slot *s = &decs[tok >> 8 & 0xff].slot[tok & 0xff];
        if (s->state == S_WAIT && !s->fence_done)
            for (int k = 0; k < WAITERS; k++)
                if (!s->waiter[k] || s->waiter[k] == ev) {
                    s->waiter[k] = ev, now = 0;
                    break;
                }
        plat_mutex_unlock(&lock);
    }
    TRACE("sync point request %#x: %s", tok, now ? "signalled at once" : "waits for its picture");
    if (now) kern_signal_system_event(ev);
}
// void CancelSyncPointEvent(SyncPointEventType *): the request is dropped and the event cleared
static void ov_sp_cancel(Cpu *c) {
    uint64_t ev = event_for(c, arg_x(c, 0));
    plat_mutex_lock(&lock);
    for (int k = 0; k < DECODERS; k++)
        for (int i = 0; i < SLOTS; i++)
            for (int j = 0; j < WAITERS; j++)
                if (decs[k].slot[i].waiter[j] == ev) decs[k].slot[i].waiter[j] = 0;
    plat_mutex_unlock(&lock);
    clear_event(c, ev);
}
// void DestroySyncPointEvent(SyncPointEventType *): the event stays bound for the next Create on the same object
static void ov_sp_nop(Cpu *c) { (void)c; }

// LOCAL TEST (Ocarina): AV1 shares the VP9 decoder's slots and events, and decodes with dav1d.
static void av1_init(Cpu *c, uint64_t key, unsigned w, unsigned h) {
    vp9_init(c, key, w, h);
    plat_mutex_lock(&lock);
    Dec *d = dec_of(key);
    Dav1dSettings set;
    dav1d_default_settings(&set);
    // OOT_AV1_THREADS=<n>: dav1d threads (default 4; 0 lets dav1d pick from the core count).
    const char *th = getenv("OOT_AV1_THREADS");
    set.n_threads = th ? atoi(th) : 4;
    set.max_frame_delay = 1;   // a picture per packet: the player waits for each one
    if (d && dav1d_open(&d->av1, &set) < 0) d->av1 = NULL;
    plat_mutex_unlock(&lock);
    rt_log("codec: AV1 decoder %ux%u (dav1d %s)%s", w, h, dav1d_version(), d && d->av1 ? "" : " failed to start: pictures stay black");
}
static void ov_av1_init(Cpu *c) { av1_init(c, arg_x(c, 0), arg_x(c, 4) & 0xffff, arg_x(c, 5) & 0xffff); }
static void ov_av1_init_remote(Cpu *c) { av1_init(c, arg_x(c, 0), arg_x(c, 2) & 0xffff, arg_x(c, 3) & 0xffff); }
static void ov_av1_flush(Cpu *c) {
    plat_mutex_lock(&lock);
    Dec *d = dec_of(arg_x(c, 0));
    if (d) quiesce(d);   // the worker may be inside dav1d for this decoder
    if (d && d->av1) dav1d_flush(d->av1);
    plat_mutex_unlock(&lock);
    ov_vp9_flush(c);
}
static void ov_av1_ready_event(Cpu *c) {   // SystemEventType *GetAv1DecoderSubmissionReadyEvent(Av1DecoderType *)
    uint64_t ev = event_for(c, arg_x(c, 0));
    kern_signal_system_event(ev);
    ret_x(c, ev);
}
static void ov_av1_props(Cpu *c) {
    uint32_t i = (uint32_t)arg_x(c, 7);
    plat_mutex_lock(&lock);
    Dec *d = dec_of(arg_x(c, 6));
    Slot s = d && i < SLOTS ? d->slot[i] : (Slot){0};
    plat_mutex_unlock(&lock);
    for (int k = 0; k < 4; k++) {
        uint16_t *p = arg_p(c, k);
        if (p) *p = k & 1 ? s.h : s.w;
    }
    if (arg_p(c, 4)) *(int64_t *)arg_p(c, 4) = s.pts;
    if (arg_p(c, 5)) *(uint8_t *)arg_p(c, 5) = (uint8_t)s.eos;
    ret_x(c, d ? 0 : R_NOT_READY);
}
HOST_OVERRIDE("_ZN2nn5codec36InitializeAv1DecoderByTransferMemoryEPNS0_14Av1DecoderTypeEPvmmttjNS0_23VideoDecoderInputFormatENS0_24VideoDecoderOutputFormatENS0_22VideoDecoderOutputModeE", ov_av1_init);
HOST_OVERRIDE("_ZN2nn5codec34InitializeAv1DecoderByRemoteMemoryEPNS0_14Av1DecoderTypeEmttjNS0_23VideoDecoderInputFormatENS0_24VideoDecoderOutputFormatENS0_22VideoDecoderOutputModeE", ov_av1_init_remote);
HOST_OVERRIDE("_ZN2nn5codec18FinalizeAv1DecoderEPNS0_14Av1DecoderTypeE", ov_vp9_fini);
HOST_OVERRIDE("_ZN2nn5codec15StartAv1DecoderEPNS0_14Av1DecoderTypeE", ov_ok);
HOST_OVERRIDE("_ZN2nn5codec14StopAv1DecoderEPNS0_14Av1DecoderTypeE", ov_ok);
HOST_OVERRIDE("_ZN2nn5codec15FlushAv1DecoderEPNS0_14Av1DecoderTypeE", ov_av1_flush);
HOST_OVERRIDE("_ZN2nn5codec24RegisterAv1DecoderBufferEPNS0_14Av1DecoderTypeEjPNS0_25VideoDecoderImagePoolTypeEj", ov_vp9_register);
HOST_OVERRIDE("_ZN2nn5codec27SubmitAv1DecoderInputBufferEPmPNS0_14Av1DecoderTypeEPKvmNS_8TimeSpanEb", ov_vp9_submit);
HOST_OVERRIDE("_ZN2nn5codec29AcquireAv1DecoderOutputBufferEPjPNS_2gs16NativeSyncHandleEPNS0_14Av1DecoderTypeE", ov_vp9_acquire);
HOST_OVERRIDE("_ZN2nn5codec29ReleaseAv1DecoderOutputBufferEPNS0_14Av1DecoderTypeEjNS_2gs16NativeSyncHandleE", ov_vp9_release);
HOST_OVERRIDE("_ZN2nn5codec35GetAv1DecoderOutputBufferPropertiesEPtS1_S1_S1_PNS_8TimeSpanEPbPNS0_14Av1DecoderTypeEj", ov_av1_props);
HOST_OVERRIDE("_ZN2nn5codec33GetAv1DecoderSubmissionReadyEventEPNS0_14Av1DecoderTypeE", ov_av1_ready_event);
HOST_OVERRIDE("_ZN2nn5codec26InitializeDecoderImagePoolEPNS0_25VideoDecoderImagePoolTypeEPvmRKNS0_27VideoDecoderImagePreferenceEj", ov_pool_init);
HOST_OVERRIDE("_ZN2nn5codec5video10InitializeEv", ov_video_init);
HOST_OVERRIDE("_ZN2nn5codec5video8FinalizeEv", ov_video_fini);
HOST_OVERRIDE("_ZN2nn5codec20InitializeVp9DecoderEPNS0_14Vp9DecoderTypeEPvmmttjNS0_23VideoDecoderInputFormatENS0_"
              "24VideoDecoderOutputFormatE", ov_vp9_init);
HOST_OVERRIDE("_ZN2nn5codec20InitializeVp9DecoderEPNS0_14Vp9DecoderTypeEmmttjNS0_23VideoDecoderInputFormatENS0_"
              "24VideoDecoderOutputFormatE", ov_vp9_init_nowork);
HOST_OVERRIDE("_ZN2nn5codec18FinalizeVp9DecoderEPNS0_14Vp9DecoderTypeE", ov_vp9_fini);
HOST_OVERRIDE("_ZN2nn5codec15StartVp9DecoderEPNS0_14Vp9DecoderTypeE", ov_ok);
HOST_OVERRIDE("_ZN2nn5codec14StopVp9DecoderEPNS0_14Vp9DecoderTypeE", ov_ok);
HOST_OVERRIDE("_ZN2nn5codec15FlushVp9DecoderEPNS0_14Vp9DecoderTypeE", ov_vp9_flush);
HOST_OVERRIDE("_ZN2nn5codec24RegisterVp9DecoderBufferEPNS0_14Vp9DecoderTypeEjPNS0_25VideoDecoderImagePoolTypeEj",
              ov_vp9_register);
HOST_OVERRIDE("_ZN2nn5codec26InitializeDecoderImagePoolEPNS0_25VideoDecoderImagePoolTypeEPvmRKNS0_27VideoDecoderImage"
              "PreferenceEjNS0_9CodecModeE", ov_pool_init);
HOST_OVERRIDE("_ZN2nn5codec24FinalizeDecoderImagePoolEPNS0_25VideoDecoderImagePoolTypeE", ov_ok);
HOST_OVERRIDE("_ZN2nn5codec31CalculateDecoderImagePoolBufferEPmS1_RKNS0_27VideoDecoderImagePreferenceEj", ov_calc_size);
HOST_OVERRIDE("_ZN2nn5codec29CalculateVp9DecoderWorkBufferEPmS1_mtt", ov_calc_size);
HOST_OVERRIDE("_ZN2nn5codec33GetVp9DecoderSubmissionReadyEventEPPNS_2os15SystemEventTypeEPNS0_14Vp9DecoderTypeE",
              ov_vp9_ready_event);
HOST_OVERRIDE("_ZN2nn5codec27SubmitVp9DecoderInputBufferEPmPNS0_14Vp9DecoderTypeEPKvmNS_8TimeSpanEb", ov_vp9_submit);
HOST_OVERRIDE("_ZN2nn5codec29AcquireVp9DecoderOutputBufferEPjPNS_2gs16NativeSyncHandleEPNS0_14Vp9DecoderTypeE",
              ov_vp9_acquire);
HOST_OVERRIDE("_ZN2nn5codec29ReleaseVp9DecoderOutputBufferEPNS0_14Vp9DecoderTypeEjNS_2gs16NativeSyncHandleE",
              ov_vp9_release);
HOST_OVERRIDE("_ZN2nn5codec35GetVp9DecoderOutputBufferPropertiesEPtS1_S1_S1_PNS_5movie10ColorSpaceEPNS_8TimeSpanEPbPNS0_"
              "14Vp9DecoderTypeEj", ov_vp9_props);
HOST_OVERRIDE("_ZN2nn5codec26ReadDecoderOutputImageNv12EPKNS0_25VideoDecoderImagePoolTypeEjttttPvmmS4_mm", ov_read_nv12);
HOST_OVERRIDE("_ZN2nn5codec36GetDecoderImagePoolNativeImageHandleEPKNS0_25VideoDecoderImagePoolTypeEj", ov_pool_image);
HOST_OVERRIDE("_ZN2nn6syncpt3sdk20CreateSyncPointEventEPNS1_18SyncPointEventTypeE", ov_sp_create);
HOST_OVERRIDE("_ZN2nn6syncpt3sdk25InitializeMultiWaitHolderEPNS_2os19MultiWaitHolderTypeEPNS1_18SyncPointEventTypeE",
              ov_sp_holder);
HOST_OVERRIDE("_ZN2nn6syncpt3sdk21TryWaitSyncPointEventEPKNS1_18SyncPointEventTypeE", ov_sp_trywait);
HOST_OVERRIDE("_ZN2nn6syncpt3sdk21RequestSyncPointEventEPNS1_18SyncPointEventTypeERKNS1_15SyncPointHandleE",
              ov_sp_request);
HOST_OVERRIDE("_ZN2nn6syncpt3sdk20CancelSyncPointEventEPNS1_18SyncPointEventTypeE", ov_sp_cancel);
HOST_OVERRIDE("_ZN2nn6syncpt3sdk21DestroySyncPointEventEPNS1_18SyncPointEventTypeE", ov_sp_nop);

// LOCAL TEST (Ocarina): a decoded picture's post sync has always passed here (decoding is finished on return).
// bool TryWaitNativeSync(const NativeSyncHandle &): a decoder fence is signalled once its buffer got a picture, anything
// else at once. Answering 1 for a buffer that waits for a picture never ended movie::VideoDecoder::Flush: its drain loop
// releases a frame, reserves a new buffer and asks again, so stopping a movie hung the game (Ocarina's New Game).
static void ov_try_wait_native_sync(Cpu *c) {
    const uint32_t *h = arg_p(c, 0);
    uint32_t tok = h ? *h : 0;
    int done = 1;
    if ((tok & 0xffff0000u) == FENCE_TAG && (tok >> 8 & 0xff) < DECODERS && (tok & 0xff) < SLOTS) {
        plat_mutex_lock(&lock);
        const Slot *s = &decs[tok >> 8 & 0xff].slot[tok & 0xff];
        done = s->state != S_WAIT || s->fence_done;
        plat_mutex_unlock(&lock);
    }
    ret_x(c, done);
}
HOST_OVERRIDE("_ZN2nn2gs17TryWaitNativeSyncERKNS0_16NativeSyncHandleE", ov_try_wait_native_sync);
