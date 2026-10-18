import { describe, expect, it } from 'vitest';
import { decodeGif } from '../../test/gifDecode';
import { encodeGif, lzwEncode, palette332, quantize332, quantizeFrame } from './gif';

/*
 * The GIF89a encoder (gif.ts), checked by decoding what it writes with an independent decoder (test/gifDecode.ts): the frame count, the size,
 * the delay, the loop and every pixel come back as asked, including frames noisy enough to fill the 12-bit table and force clear codes.
 */

function rgbaOf(width: number, height: number, colourAt: (x: number, y: number) => [number, number, number]): Uint8ClampedArray {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const [r, g, b] = colourAt(x, y);
      data.set([r, g, b, 255], (y * width + x) * 4);
    }
  }
  return data;
}

function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

describe('the 3-3-2 palette', () => {
  it('gives each of the 256 colours an index that maps back to itself', () => {
    for (let i = 0; i < 256; i += 1) {
      const [r, g, b] = palette332(i);
      expect(quantize332(r, g, b)).toBe(i);
    }
  });

  it('puts red in the high three bits, green in the next three and blue in the low two', () => {
    expect(quantize332(255, 0, 0)).toBe(0b11100000);
    expect(quantize332(0, 255, 0)).toBe(0b00011100);
    expect(quantize332(0, 0, 255)).toBe(0b00000011);
    expect(quantize332(0, 0, 0)).toBe(0);
    expect(quantize332(255, 255, 255)).toBe(255);
  });
});

describe('encodeGif', () => {
  it('writes a GIF89a that decodes to the size, the frame count, the delay and the loop that were asked for', () => {
    const frames = [0, 1, 2, 3, 4].map((n) => ({ rgba: rgbaOf(7, 5, (x, y) => [(x * 30 + n * 40) & 255, (y * 50) & 255, n * 50]) }));
    const gif = decodeGif(encodeGif(frames, { width: 7, height: 5, delayCentiseconds: 10 }));
    expect([gif.width, gif.height]).toEqual([7, 5]);
    expect(gif.frames).toHaveLength(5);
    expect(gif.delays).toEqual([10, 10, 10, 10, 10]);
    expect(gif.loops).toBe(true);
    expect(gif.trailer).toBe(true);
    expect(gif.palette.length).toBe(768);
  });

  it('leaves each frame in place under the next, with no transparency, from the corner, at the minimum code size of eight', () => {
    const frames = [0, 1].map(() => ({ indexes: new Uint8Array(12).fill(5) }));
    const gif = decodeGif(encodeGif(frames, { width: 4, height: 3, delayCentiseconds: 9 }));
    expect(gif.controlPacked).toEqual([4, 4]);                                  // disposal 1 (leave in place), no transparency flag
    expect(gif.transparentIndex).toEqual([0, 0]);
    expect(gif.placements).toEqual([{ left: 0, top: 0, width: 4, height: 3, packed: 0, minCodeSize: 8 }, { left: 0, top: 0, width: 4, height: 3, packed: 0, minCodeSize: 8 }]);
    expect([gif.backgroundIndex, gif.aspect]).toEqual([0, 0]);
  });

  it('puts every pixel of every frame at the palette index of its colour', () => {
    const frames = [0, 1, 2].map((n) => ({ rgba: rgbaOf(17, 13, (x, y) => [(x * 15 + n) & 255, (y * 19 + 3 * n) & 255, (x * y + n) & 255]) }));
    const gif = decodeGif(encodeGif(frames, { width: 17, height: 13, delayCentiseconds: 7 }));
    frames.forEach((f, i) => expect([...gif.frames[i]!]).toEqual([...quantizeFrame(f.rgba)]));
  });

  it('decodes a palette entry to the colour the encoder meant', () => {
    const gif = decodeGif(encodeGif([{ rgba: rgbaOf(1, 1, () => [255, 0, 0]) }], { width: 1, height: 1, delayCentiseconds: 1 }));
    const index = gif.frames[0]![0]!;
    expect([...gif.palette.subarray(index * 3, index * 3 + 3)]).toEqual([255, 0, 0]);
  });

  it('survives noise that fills the code table and forces clear codes', () => {
    const rand = seeded(5);
    const noise = rgbaOf(160, 90, () => [rand() * 256, rand() * 256, rand() * 256]);
    const flat = rgbaOf(160, 90, () => [11, 16, 14]);
    const gif = decodeGif(encodeGif([{ rgba: noise }, { rgba: flat }, { rgba: noise }], { width: 160, height: 90, delayCentiseconds: 3 }));
    expect(gif.frames).toHaveLength(3);
    expect([...gif.frames[0]!]).toEqual([...quantizeFrame(noise)]);
    expect([...gif.frames[1]!]).toEqual([...quantizeFrame(flat)]);
    expect([...gif.frames[2]!]).toEqual([...quantizeFrame(noise)]);
  });

  it('accepts frames already reduced to indexes, and writes the same thing', () => {
    const rgba = rgbaOf(9, 9, (x, y) => [x * 28, y * 28, 128]);
    const a = encodeGif([{ rgba }], { width: 9, height: 9, delayCentiseconds: 5 });
    const b = encodeGif([{ indexes: quantizeFrame(rgba) }], { width: 9, height: 9, delayCentiseconds: 5 });
    expect([...a]).toEqual([...b]);
  });

  it('writes a one-pixel and a one-colour frame', () => {
    const gif = decodeGif(encodeGif([{ indexes: Uint8Array.of(77) }], { width: 1, height: 1, delayCentiseconds: 2 }));
    expect([...gif.frames[0]!]).toEqual([77]);
    const flat = decodeGif(encodeGif([{ indexes: new Uint8Array(5000).fill(9) }], { width: 100, height: 50, delayCentiseconds: 2 }));
    expect(flat.frames[0]!.every((v) => v === 9)).toBe(true);
  });

  it('refuses a bad size, a bad delay, no frames and a frame of the wrong length', () => {
    const frame = { indexes: new Uint8Array(4) };
    expect(() => encodeGif([frame], { width: 0, height: 2, delayCentiseconds: 1 })).toThrow(RangeError);
    expect(() => encodeGif([frame], { width: 2, height: 70000, delayCentiseconds: 1 })).toThrow(RangeError);
    expect(() => encodeGif([frame], { width: 65536, height: 2, delayCentiseconds: 1 })).toThrow(/1 to 65535 pixels/);
    expect(() => encodeGif([frame], { width: 2.5, height: 2, delayCentiseconds: 1 })).toThrow(RangeError);
    expect(() => encodeGif([frame], { width: 2, height: 2, delayCentiseconds: 0 })).toThrow(RangeError);
    expect(() => encodeGif([frame], { width: 2, height: 2, delayCentiseconds: 70000 })).toThrow(RangeError);
    expect(() => encodeGif([], { width: 2, height: 2, delayCentiseconds: 1 })).toThrow(/at least one frame/);
    expect(() => encodeGif([{ indexes: new Uint8Array(3) }], { width: 2, height: 2, delayCentiseconds: 1 })).toThrow(/3 indexes/);
    expect(() => encodeGif([{ rgba: new Uint8ClampedArray(15) }], { width: 2, height: 2, delayCentiseconds: 1 })).toThrow(/15 bytes/);
    expect(() => encodeGif([{}], { width: 2, height: 2, delayCentiseconds: 1 })).toThrow(/0 bytes/);
  });

  it('starts with the GIF89a header and the screen size, little-endian, and ends with the trailer', () => {
    const bytes = encodeGif([{ indexes: new Uint8Array(300 * 2) }], { width: 300, height: 2, delayCentiseconds: 1 });
    expect(String.fromCharCode(...bytes.subarray(0, 6))).toBe('GIF89a');
    expect([bytes[6], bytes[7], bytes[8], bytes[9]]).toEqual([300 & 255, 300 >> 8, 2, 0]);
    expect(bytes[bytes.length - 1]).toBe(0x3b);
  });
});

describe('lzwEncode', () => {
  it('starts with a clear code and ends with the end code, and is shorter than the data for a flat frame', () => {
    const out = lzwEncode(new Uint8Array(10000).fill(3));
    expect(out.length).toBeLessThan(500);
    expect(out[0]).toBe(0x00);                      // clear code 256 = 100000000b, 9 bits, low bits first
    expect(out[1]! & 1).toBe(1);
  });
});
