/**
 * Table cell filling: puts a region's OCR lines into the `<tr>`/`<td>`
 * skeleton the structure decoder produced, by clustering the line centers
 * into the skeleton's rows and columns.
 */
import { boundingBoxOfPoints } from 'react-native-executorch/cv';
import type { OcrDetection } from '../ocr/types';

/**
 * Clusters 1-D values into `k` groups at the `k - 1` widest gaps and reduces
 * each group to its mean. Table cells distribute that way: tight within a row
 * or column, with gaps between.
 * @param values The coordinates to cluster.
 * @param k The number of groups.
 * @returns The group centers in ascending order; fewer than `k` when there
 * are fewer values.
 */
export function clusterByGaps(values: readonly number[], k: number): number[] {
  'worklet';
  if (k < 1) {
    return [];
  }
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length <= k) {
    return sorted;
  }
  const gaps = sorted.slice(1).map((value, i) => ({ at: i + 1, size: value - sorted[i]! }));
  gaps.sort((a, b) => b.size - a.size);
  const cuts = gaps
    .slice(0, k - 1)
    .map((gap) => gap.at)
    .sort((a, b) => a - b);
  const centers: number[] = [];
  let prev = 0;
  for (const cut of [...cuts, sorted.length]) {
    const group = sorted.slice(prev, cut);
    centers.push(group.reduce((sum, value) => sum + value, 0) / group.length);
    prev = cut;
  }
  return centers;
}

/**
 * Index of the center nearest to `value`.
 * @param value The coordinate to place.
 * @param centers The cluster centers.
 * @returns The index of the nearest center (0 when there are none).
 */
function nearestCenter(value: number, centers: readonly number[]): number {
  'worklet';
  let best = 0;
  for (let i = 1; i < centers.length; i++) {
    if (Math.abs(value - centers[i]!) < Math.abs(value - centers[best]!)) {
      best = i;
    }
  }
  return best;
}

/**
 * Fills a table skeleton's cells with a region's OCR lines: each line's box
 * center is assigned to its nearest row and column cluster, and lines that
 * land in the same cell are joined with spaces. Without a usable grid (no
 * rows, no columns or no lines) the cells are filled in document order.
 *
 * Known limit: a line whose center falls inside a rowspan's vertical extent
 * lands on a row where that column has no `<td>` (the span occupies it), so
 * its text is dropped. Occupancy is tracked; routing into spanned cells is not.
 * @category Functions
 * @param html The structure HTML with empty `<td></td>` cells, as the
 * structure decoder emits it.
 * @param lines The region's OCR lines, in the same frame as each other.
 * @returns The HTML with the cell texts filled in.
 */
export function fillTableCells(html: string, lines: readonly OcrDetection[]): string {
  'worklet';
  const rowCount = (html.match(/<tr>/g) ?? []).length;
  let colCount = 0;
  const rowRegex = /<tr>([\s\S]*?)<\/tr>/g;
  let row: RegExpExecArray | null;
  while ((row = rowRegex.exec(html)) !== null) {
    let cols = 0;
    for (const td of row[1]!.match(/<td[^>]*>/g) ?? []) {
      cols += Number((td.match(/colspan="(\d+)"/) ?? [])[1] ?? 1);
    }
    colCount = Math.max(colCount, cols);
  }
  if (rowCount === 0 || colCount === 0 || lines.length === 0) {
    let cell = 0;
    const texts = lines.map((l) => l.text);
    // Matches empty `<td…></td>` cells only: the decoder emits the fused
    // empty-cell token, which the SLANet vocab has. A vocab that emitted
    // pre-filled cells would need this widened.
    return html.replace(/<td([^>]*)><\/td>/g, (_match, attrs: string) => {
      const text = cell < texts.length ? texts[cell]! : '';
      cell++;
      return `<td${attrs}>${text}</td>`;
    });
  }

  const centersX: number[] = [];
  const centersY: number[] = [];
  for (const line of lines) {
    const box = boundingBoxOfPoints(line.quad, 'xyxy');
    centersX.push((box.xmin + box.xmax) / 2);
    centersY.push((box.ymin + box.ymax) / 2);
  }
  const rowCenters = clusterByGaps(centersY, rowCount);
  const colCenters = clusterByGaps(centersX, colCount);
  const grid: string[][] = Array.from({ length: rowCenters.length }, () =>
    new Array<string>(colCenters.length).fill('')
  );
  for (let i = 0; i < lines.length; i++) {
    const r = nearestCenter(centersY[i]!, rowCenters);
    const c = nearestCenter(centersX[i]!, colCenters);
    grid[r]![c] = `${grid[r]![c]!} ${lines[i]!.text}`.trim();
  }

  const occupied: boolean[][] = [];
  let rowIdx = -1;
  return html.replace(/<tr>|<td([^>]*)><\/td>/g, (match, attrs: string | undefined) => {
    if (attrs === undefined) {
      rowIdx++;
      return match;
    }
    let c = 0;
    while (occupied[rowIdx]?.[c]) {
      c++;
    }
    const colspan = Number((attrs.match(/colspan="(\d+)"/) ?? [])[1] ?? 1);
    const rowspan = Number((attrs.match(/rowspan="(\d+)"/) ?? [])[1] ?? 1);
    for (let dr = 0; dr < rowspan; dr++) {
      const rr = rowIdx + dr;
      occupied[rr] ??= [];
      for (let dc = 0; dc < colspan; dc++) {
        occupied[rr]![c + dc] = true;
      }
    }
    return `<td${attrs}>${grid[rowIdx]?.[c] ?? ''}</td>`;
  });
}
