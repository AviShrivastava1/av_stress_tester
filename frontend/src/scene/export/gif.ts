/*
 * A small GIF89a encoder, written for this repository so that the scene export needs no new dependency. Behind the Waymo-derived outputs
 * switch (waymoDerivedOutputs.ts): nothing in the app reaches it while the switch is off.
 *
 * What it writes: the header and logical screen descriptor, one global colour table of 256 entries, the NETSCAPE2.0 looping extension (loop forever),
 * and for every frame a graphic control extension (delay in hundredths of a second), an image descriptor and LZW-compressed colour indexes in
 * sub-blocks, then the trailer. Colours are reduced to a fixed 3-3-2 palette (3 bits of red, 3 of green, 2 of blue): the same 256 colours for every
 * frame, so there is no per-frame table to write and no state to carry. This is a coarse palette for a scene of flat colours, not a photograph encoder.
 * Transparency is not used: alpha is ignored (a pixel is its colour).
 *
 * The LZW coder follows the usual GIF variant (initial code size 9 for a minimum of 8, clear code 256, end code 257, codes up to 12 bits, a
 * clear code when the table is full). gif.test.ts decodes what it writes with an independent decoder and checks the frame count, the size and
 * every pixel.
 */

export interface GifFrame {
  /** width * height * 4 bytes, RGBA, row by row from the top left (the layout of ImageData.data). Either this or `indexes`. */
  rgba?: Uint8ClampedArray | Uint8Array;
  /** width * height palette indexes (see quantizeFrame): a frame already reduced, which takes a quarter of the memory. Either this or `rgba`. */
  indexes?: Uint8Array;
}

export interface GifOptions {
  width: number;
  height: number;
  /** Delay between frames in hundredths of a second (1 to 65535). */
  delayCentiseconds: number;
}

/** The index in the 3-3-2 palette of an RGB colour: red in the high three bits, green in the next three, blue in the low two. */
export function quantize332(r: number, g: number, b: number): number {
  return ((r >> 5) << 5) | ((g >> 5) << 2) | (b >> 6);
}

/** The RGB colour the 3-3-2 palette has at an index: each channel spread over 0 to 255. */
export function palette332(index: number): [number, number, number] {
  return [Math.round(((index >> 5) & 7) * 255 / 7), Math.round(((index >> 2) & 7) * 255 / 7), Math.round((index & 3) * 255 / 3)];
}

/** The 3-3-2 palette index of every pixel of an RGBA frame. */
export function quantizeFrame(rgba: Uint8ClampedArray | Uint8Array): Uint8Array {
  const indexes = new Uint8Array(Math.floor(rgba.length / 4));
  for (let p = 0, q = 0; p < indexes.length; p += 1, q += 4) indexes[p] = quantize332(rgba[q]!, rgba[q + 1]!, rgba[q + 2]!);
  return indexes;
}

function checkSize(width: number, height: number): void {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width > 65535 || height > 65535) {
    throw new RangeError(`a GIF is 1 to 65535 pixels each way, not ${width} by ${height}`);
  }
}

class ByteSink {
  private bytes: number[] = [];
  push(...values: number[]): void {
    for (const v of values) this.bytes.push(v);
  }
  u16(v: number): void {
    this.bytes.push(v & 255, (v >> 8) & 255);
  }
  ascii(text: string): void {
    for (let i = 0; i < text.length; i += 1) this.bytes.push(text.charCodeAt(i));
  }
  append(more: Uint8Array): void {
    for (let i = 0; i < more.length; i += 1) this.bytes.push(more[i]!);
  }
  done(): Uint8Array {
    return Uint8Array.from(this.bytes);
  }
}

/** LZW-compress colour indexes (0 to 255) with a minimum code size of 8; returns the code stream as bytes, not yet cut into sub-blocks. */
export function lzwEncode(indexes: Uint8Array): Uint8Array {
  const MIN = 8;
  const CLEAR = 1 << MIN;
  const END = CLEAR + 1;
  const out: number[] = [];
  let acc = 0;
  let nbits = 0;
  let codeSize = MIN + 1;
  let nextCode = END + 1;
  let table = new Map<number, number>();

  const emit = (code: number): void => {
    acc |= code << nbits;
    nbits += codeSize;
    while (nbits >= 8) {
      out.push(acc & 255);
      acc >>>= 8;
      nbits -= 8;
    }
  };

  emit(CLEAR);
  let prefix = indexes[0]!;
  for (let i = 1; i < indexes.length; i += 1) {
    const k = indexes[i]!;
    const key = (prefix << 8) | k;
    const found = table.get(key);
    if (found !== undefined) {
      prefix = found;
      continue;
    }
    emit(prefix);
    if (nextCode === 4096) {
      emit(CLEAR);
      nextCode = END + 1;
      codeSize = MIN + 1;
      table = new Map();
    } else {
      if (nextCode >= 1 << codeSize) codeSize += 1;
      table.set(key, nextCode);
      nextCode += 1;
    }
    prefix = k;
  }
  emit(prefix);
  emit(END);
  if (nbits > 0) out.push(acc & 255);
  return Uint8Array.from(out);
}

function subBlocks(sink: ByteSink, data: Uint8Array): void {
  for (let i = 0; i < data.length; i += 255) {
    const chunk = data.subarray(i, Math.min(i + 255, data.length));
    sink.push(chunk.length);
    sink.append(chunk);
  }
  sink.push(0);
}

/** A GIF89a of the frames, each width * height RGBA, played in order and looped for ever. Throws RangeError for a bad size, delay or frame. */
export function encodeGif(frames: GifFrame[], options: GifOptions): Uint8Array {
  const { width, height, delayCentiseconds } = options;
  checkSize(width, height);
  if (!Number.isInteger(delayCentiseconds) || delayCentiseconds < 1 || delayCentiseconds > 65535) {
    throw new RangeError(`the delay is 1 to 65535 hundredths of a second, not ${delayCentiseconds}`);
  }
  if (frames.length < 1) throw new RangeError('a GIF needs at least one frame');

  const sink = new ByteSink();
  sink.ascii('GIF89a');
  sink.u16(width);
  sink.u16(height);
  sink.push(0xf7, 0, 0);                                    // global colour table, 8 bits of colour, 256 entries; background 0; square pixels
  for (let i = 0; i < 256; i += 1) sink.push(...palette332(i));
  sink.push(0x21, 0xff, 0x0b);
  sink.ascii('NETSCAPE2.0');
  sink.push(0x03, 0x01, 0x00, 0x00, 0x00);                  // loop for ever

  for (const [n, frame] of frames.entries()) {
    let indexes: Uint8Array;
    if (frame.indexes !== undefined) {
      if (frame.indexes.length !== width * height) {
        throw new RangeError(`frame ${n} has ${frame.indexes.length} indexes, not ${width} * ${height}`);
      }
      indexes = frame.indexes;
    } else {
      if (frame.rgba === undefined || frame.rgba.length !== width * height * 4) {
        throw new RangeError(`frame ${n} has ${frame.rgba?.length ?? 0} bytes, not ${width} * ${height} * 4`);
      }
      indexes = quantizeFrame(frame.rgba);
    }
    sink.push(0x21, 0xf9, 0x04, 0x04);                      // graphic control: leave the frame in place, no transparency
    sink.u16(delayCentiseconds);
    sink.push(0x00, 0x00);
    sink.push(0x2c);
    sink.u16(0);
    sink.u16(0);
    sink.u16(width);
    sink.u16(height);
    sink.push(0x00);                                         // no local colour table, not interlaced
    sink.push(8);                                            // minimum LZW code size
    subBlocks(sink, lzwEncode(indexes));
  }
  sink.push(0x3b);
  return sink.done();
}
