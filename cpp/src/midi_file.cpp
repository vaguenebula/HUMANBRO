// Standard MIDI File reader with the same note semantics as midi_io.parse_midi (mido):
// note_on velocity>0 starts a note, note_off / note_on velocity 0 ends it, a re-strike of a
// held key ends the previous instance, unreleased keys end at the end of their track, and
// channel 10 (drums) is skipped. Running status follows mido (meta events do not set it).
// Every event is also kept per track, so a file can be rewritten with re-timed notes
// (save_with_timing, the counterpart of midi_io.with_timing).
#include <algorithm>
#include <fstream>
#include <map>
#include <numeric>

#include "humanbro/humanbro.hpp"

namespace humanbro {
namespace {

constexpr int kDrumChannel = 9;

class Cursor {
public:
    Cursor(const std::vector<uint8_t>& b, std::size_t pos, std::size_t end) : b_(b), pos_(pos), end_(end) {}
    bool done() const { return pos_ >= end_; }
    std::size_t pos() const { return pos_; }
    uint8_t byte() {
        if (pos_ >= end_) throw Error("unexpected end of MIDI track");
        return b_[pos_++];
    }
    uint32_t vlq() {
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            const uint8_t c = byte();
            v = (v << 7) | (c & 0x7F);
            if (!(c & 0x80)) return v;
        }
        throw Error("invalid variable-length quantity");
    }
    void skip(std::size_t n) {
        if (pos_ + n > end_) throw Error("unexpected end of MIDI track");
        pos_ += n;
    }

private:
    const std::vector<uint8_t>& b_;
    std::size_t pos_, end_;
};

uint32_t be32(const std::vector<uint8_t>& b, std::size_t p) {
    if (p + 4 > b.size()) throw Error("truncated MIDI file");
    return (uint32_t(b[p]) << 24) | (uint32_t(b[p + 1]) << 16) | (uint32_t(b[p + 2]) << 8) | b[p + 3];
}
void put_be32(std::vector<uint8_t>& out, uint32_t v) {
    for (int s = 24; s >= 0; s -= 8) out.push_back(uint8_t(v >> s));
}

void put_vlq(std::vector<uint8_t>& out, uint64_t v) {
    if (v > 0x0FFFFFFF) throw Error("delta time too large for a MIDI file");
    uint8_t buf[4];
    int n = 0;
    buf[n++] = uint8_t(v & 0x7F);
    while (v >>= 7) buf[n++] = uint8_t((v & 0x7F) | 0x80);
    while (n) out.push_back(buf[--n]);
}

uint16_t be16(const std::vector<uint8_t>& b, std::size_t p) {
    if (p + 2 > b.size()) throw Error("truncated MIDI file");
    return uint16_t((b[p] << 8) | b[p + 1]);
}

int message_size(uint8_t status) {  // including the status byte (mido's spec table)
    switch (status & 0xF0) {
        case 0x80: case 0x90: case 0xA0: case 0xB0: case 0xE0: return 3;
        case 0xC0: case 0xD0: return 2;
        default: break;
    }
    switch (status) {
        case 0xF1: case 0xF3: return 2;
        case 0xF2: return 3;
        case 0xF6: case 0xF8: case 0xFA: case 0xFB: case 0xFC: case 0xFE: return 1;
        default: throw Error("undefined MIDI status byte");
    }
}

}  // namespace

MidiFile MidiFile::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error("cannot open MIDI file: " + path);
    MidiFile f;
    f.bytes_.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    const auto& b = f.bytes_;
    if (b.size() < 14 || std::string(b.begin(), b.begin() + 4) != "MThd") throw Error("not a MIDI file: " + path);
    const uint32_t header_len = be32(b, 4);
    const uint16_t format = be16(b, 8), n_tracks = be16(b, 10), division = be16(b, 12);
    if (format == 2) throw Error("type-2 (asynchronous) MIDI files are not supported");
    if (division & 0x8000 || division == 0) throw Error("SMPTE / invalid time division is not supported");
    f.format_ = format;
    f.division_ = division;
    f.tracks_.resize(n_tracks);
    f.track_end_.assign(n_tracks, -1);

    Score& s = f.score_;
    s.ticks_per_quarter = division;
    std::size_t pos = 8 + header_len;
    int64_t end_tick = 0;
    for (uint16_t t = 0; t < n_tracks; ++t) {
        if (pos + 8 > b.size() || std::string(b.begin() + pos, b.begin() + pos + 4) != "MTrk")
            throw Error("no MTrk header at start of track");
        const std::size_t len = be32(b, pos + 4), start = pos + 8, stop = start + len;
        if (stop > b.size()) throw Error("truncated MIDI track");
        Cursor c(b, start, stop);
        int64_t tick = 0;
        int last_status = -1;
        std::map<std::pair<int, int>, std::size_t> open;  // (channel, pitch) -> note index
        std::vector<Event>& events = f.tracks_[t];
        while (!c.done()) {
            tick += c.vlq();
            const std::size_t ev_pos = c.pos();
            int status = c.byte();
            int peek = -1;
            if (status < 0x80) {
                if (last_status < 0) throw Error("running status without previous status");
                peek = status;
                status = last_status;
            } else if (status != 0xFF) {
                last_status = status;  // meta events do not set running status (as in mido)
            }
            if (status == 0xFF) {
                const uint8_t type = c.byte();
                const uint32_t n = c.vlq();
                const std::size_t data = c.pos();
                c.skip(n);
                if (type == 0x51 && n >= 3) {
                    s.tempo_changes.push_back({tick, double((b[data] << 16) | (b[data + 1] << 8) | b[data + 2])});
                } else if (type == 0x58 && n >= 2) {
                    if (b[data] > 0 && b[data + 1] < 31) s.time_signatures.push_back({tick, b[data], 1 << b[data + 1]});
                }
                if (type == 0x2F) {  // end of track: re-added at the end when rewriting
                    f.track_end_[t] = std::max(f.track_end_[t], tick);
                } else {
                    events.push_back({tick, std::vector<uint8_t>(b.begin() + ev_pos, b.begin() + c.pos()), 0});
                }
            } else if (status == 0xF0 || status == 0xF7) {
                c.skip(c.vlq());
                events.push_back({tick, std::vector<uint8_t>(b.begin() + ev_pos, b.begin() + c.pos()), 2});
            } else {
                const int n_data = message_size(uint8_t(status)) - 1;
                int data[2] = {0, 0};
                std::size_t data_pos[2] = {0, 0};
                for (int k = 0; k < n_data; ++k) {
                    if (k == 0 && peek >= 0) {
                        data[0] = peek;
                        data_pos[0] = c.pos() - 1;
                    } else {
                        data_pos[k] = c.pos();
                        data[k] = c.byte();
                    }
                    if (data[k] > 127) data[k] = 127;  // mido clip=True
                }
                const int kind = status & 0xF0, channel = status & 0x0F;
                const bool release = kind == 0x80 || (kind == 0x90 && data[1] == 0);
                Event ev{tick, {uint8_t(status)}, uint8_t(release ? 1 : kind == 0x90 ? 3 : 2)};
                for (int k = 0; k < n_data; ++k) ev.bytes.push_back(uint8_t(data[k]));
                const std::size_t ev_index = events.size();
                events.push_back(std::move(ev));
                if ((kind == 0x90 || kind == 0x80) && channel != kDrumChannel) {
                    const auto key = std::make_pair(channel, data[0]);
                    auto it = open.find(key);
                    if (it != open.end()) {  // note-off, or re-strike of a held key
                        s.notes[it->second].offset_tick = tick;
                        if (release) f.off_event_[it->second] = static_cast<long>(ev_index);
                        open.erase(it);
                    }
                    if (kind == 0x90 && data[1] > 0) {
                        open[key] = s.notes.size();
                        Note note{tick, -1, data[0], data[1]};
                        note.channel = channel;
                        note.track = t;
                        s.notes.push_back(note);
                        f.velocity_pos_.push_back(data_pos[1]);
                        f.on_event_.push_back(ev_index);
                        f.off_event_.push_back(-1);
                    }
                }
            }
        }
        for (const auto& [key, idx] : open) s.notes[idx].offset_tick = tick;  // never released
        end_tick = std::max(end_tick, tick);
        pos = stop;
    }
    for (const Note& note : s.notes) end_tick = std::max(end_tick, std::max(note.offset_tick, note.onset_tick));
    s.end_tick = end_tick;
    return f;
}

void MidiFile::save_with_velocities(const std::string& path, const std::vector<int>& velocities) const {
    if (velocities.size() != score_.notes.size())
        throw Error("expected " + std::to_string(score_.notes.size()) + " velocities");
    std::vector<uint8_t> out = bytes_;
    for (std::size_t i = 0; i < velocities.size(); ++i)
        out[velocity_pos_[i]] = uint8_t(std::clamp(velocities[i], 1, 127));  // never 0 (= note-off)
    std::ofstream o(path, std::ios::binary);
    if (!o) throw Error("cannot write " + path);
    o.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
}

void MidiFile::save_with_timing(const std::string& path, const std::vector<int64_t>& onset_tick,
                                const std::vector<int64_t>& offset_tick, const std::vector<int>& velocities) const {
    const std::size_t n = score_.notes.size();
    if (onset_tick.size() != n || offset_tick.size() != n || (!velocities.empty() && velocities.size() != n))
        throw Error("expected one onset, offset (and velocity) per note");
    std::vector<std::vector<Event>> tracks = tracks_;
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<Event>& events = tracks[static_cast<std::size_t>(score_.notes[i].track)];
        Event& on = events[on_event_[i]];
        on.tick = std::max<int64_t>(onset_tick[i], 0);
        if (!velocities.empty()) on.bytes[2] = uint8_t(std::clamp(velocities[i], 1, 127));
        if (off_event_[i] >= 0)
            events[static_cast<std::size_t>(off_event_[i])].tick = std::max<int64_t>(offset_tick[i], 0);
    }

    std::vector<uint8_t> out = {0x4D, 0x54, 0x68, 0x64, 0, 0, 0, 6};  // "MThd", header length 6
    for (uint16_t v : {format_, static_cast<uint16_t>(tracks.size()), division_}) {
        out.push_back(uint8_t(v >> 8));
        out.push_back(uint8_t(v & 0xFF));
    }
    for (std::size_t t = 0; t < tracks.size(); ++t) {
        const std::vector<Event>& events = tracks[t];
        // Same-tick order: meta, note releases, other channel events, note starts; then file order.
        std::vector<std::size_t> idx(events.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
            const Event &x = events[a], &y = events[b];
            return x.tick != y.tick ? x.tick < y.tick : x.order_class < y.order_class;
        });
        std::vector<uint8_t> body;
        int64_t last = 0;
        for (std::size_t i : idx) {
            put_vlq(body, static_cast<uint64_t>(events[i].tick - last));
            body.insert(body.end(), events[i].bytes.begin(), events[i].bytes.end());
            last = events[i].tick;
        }
        const int64_t end = std::max(track_end_[t], last);
        put_vlq(body, static_cast<uint64_t>(end - last));
        body.insert(body.end(), {0xFF, 0x2F, 0x00});
        out.insert(out.end(), {0x4D, 0x54, 0x72, 0x6B});  // "MTrk"
        put_be32(out, static_cast<uint32_t>(body.size()));
        out.insert(out.end(), body.begin(), body.end());
    }
    std::ofstream o(path, std::ios::binary);
    if (!o) throw Error("cannot write " + path);
    o.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
}

}  // namespace humanbro
