/**
 * A GIF decoder for tests: written separately from scene/export/gif.ts (the encoder) so that a test which decodes the encoder's output
 * checks it against the format and not against itself. It reads what a GIF89a with a global colour table contains and returns the frames as
 * palette indexes. It is not a general decoder (no interlacing, no local colour tables, no disposal handling): it refuses what it does not read.
 */
export interface DecodedGif {
  width: number;
  height: number;
  palette: Uint8Array;
  frames: Uint8Array[];
  delays: number[];
  loops: boolean;
  trailer: boolean;
  backgroundIndex: number;
  aspect: number;
  /** The packed byte of every graphic control block, and its transparent colour index. */
  controlPacked: number[];
  transparentIndex: number[];
  /** Where every image sits and how it is flagged. */
  placements: { left: number; top: number; width: number; height: number; packed: number; minCodeSize: number }[];
}

export function lzwDecode(minCodeSize: number, data: Uint8Array, expected: number): Uint8Array {
  const clear = 1 << minCodeSize;
  const end = clear + 1;
  const prefix = new Int32Array(4096).fill(-1);
  const suffix = new Uint8Array(4096);
  const out = new Uint8Array(expected);
  let n = 0;
  let size = minCodeSize + 1;
  let next = end + 1;
  let prev = -1;
  let bit = 0;

  const read = (): number => {
    let code = 0;
    for (let i = 0; i < size; i += 1) {
      const byte = data[(bit + i) >> 3];
      if (byte === undefined) throw new Error('the code stream ends inside a code');
      code |= ((byte >> ((bit + i) & 7)) & 1) << i;
    }
    bit += size;
    return code;
  };
  const expand = (code: number): number[] => {
    const bytes: number[] = [];
    let c = code;
    while (c >= clear) {
      bytes.push(suffix[c]!);
      c = prefix[c]!;
    }
    bytes.push(c);
    return bytes.reverse();
  };

  for (;;) {
    const code = read();
    if (code === clear) {
      size = minCodeSize + 1;
      next = end + 1;
      prev = -1;
      continue;
    }
    if (code === end) break;
    let entry: number[];
    if (prev === -1) {
      if (code >= clear) throw new Error('the first code after a clear is not a literal');
      entry = [code];
    } else if (code < next) {
      entry = expand(code);
    } else if (code === next) {
      const before = expand(prev);
      entry = [...before, before[0]!];
    } else {
      throw new Error(`code ${code} is past the table (${next})`);
    }
    for (const b of entry) {
      if (n >= expected) throw new Error('more pixels than the frame has');
      out[n] = b;
      n += 1;
    }
    if (prev !== -1 && next < 4096) {
      prefix[next] = prev;
      suffix[next] = entry[0]!;
      next += 1;
    }
    if (next === 1 << size && size < 12) size += 1;
    prev = code;
  }
  if (n !== expected) throw new Error(`the frame decoded to ${n} pixels, not ${expected}`);
  return out;
}

export function decodeGif(bytes: Uint8Array): DecodedGif {
  let p = 0;
  const text = (len: number): string => String.fromCharCode(...bytes.subarray(p, p + len));
  const u16 = (at: number): number => bytes[at]! | (bytes[at + 1]! << 8);
  if (text(6) !== 'GIF89a') throw new Error('not a GIF89a');
  p = 6;
  const width = u16(p);
  const height = u16(p + 2);
  const packed = bytes[p + 4]!;
  p += 7;
  if ((packed & 0x80) === 0) throw new Error('no global colour table');
  const tableSize = 3 * (1 << ((packed & 7) + 1));
  const backgroundIndex = bytes[11]!;
  const aspect = bytes[12]!;
  const palette = bytes.slice(p, p + tableSize);
  p += tableSize;
  const frames: Uint8Array[] = [];
  const delays: number[] = [];
  let loops = false;
  let trailer = false;
  const controlPacked: number[] = [];
  const transparentIndex: number[] = [];
  const placements: DecodedGif['placements'] = [];
  while (p < bytes.length) {
    const block = bytes[p]!;
    if (block === 0x3b) {
      trailer = true;
      break;
    }
    if (block === 0x21) {
      const label = bytes[p + 1]!;
      p += 2;
      if (label === 0xf9) {
        if (bytes[p] !== 4) throw new Error('graphic control block is not 4 bytes');
        controlPacked.push(bytes[p + 1]!);
        delays.push(u16(p + 2));
        transparentIndex.push(bytes[p + 4]!);
        p += 1 + 4;
        if (bytes[p] !== 0) throw new Error('graphic control block is not terminated');
        p += 1;
      } else {
        if (label === 0xff && text(1) === '\u000b') {
          const name = String.fromCharCode(...bytes.subarray(p + 1, p + 12));
          if (name === 'NETSCAPE2.0') loops = true;
        }
        for (;;) {
          if (p >= bytes.length) throw new Error('an extension runs past the end of the file');
          const len = bytes[p]!;
          p += 1 + len;
          if (len === 0) break;
        }
      }
      continue;
    }
    if (block === 0x2c) {
      const w = u16(p + 5);
      const h = u16(p + 7);
      if (bytes[p + 9] !== 0) throw new Error('local colour tables and interlacing are not read here');
      const at = p;
      p += 10;
      const minCode = bytes[p]!;
      p += 1;
      const chunks: number[] = [];
      for (;;) {
        if (p >= bytes.length) throw new Error('the image data runs past the end of the file');
        const len = bytes[p]!;
        p += 1;
        if (len === 0) break;
        for (let i = 0; i < len; i += 1) chunks.push(bytes[p + i]!);
        p += len;
      }
      placements.push({ left: u16(at + 1), top: u16(at + 3), width: w, height: h, packed: bytes[at + 9]!, minCodeSize: minCode });
      frames.push(lzwDecode(minCode, Uint8Array.from(chunks), w * h));
      continue;
    }
    throw new Error(`unknown block 0x${block.toString(16)} at ${p}`);
  }
  return { width, height, palette, frames, delays, loops, trailer, backgroundIndex, aspect, controlPacked, transparentIndex, placements };
}
