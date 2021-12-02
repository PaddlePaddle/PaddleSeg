describe('e2e test humanseg model', () => {
    beforeAll(async () => {
        await page.goto(PATH);
    });

    it('humanseg infer and diff test', async () => {
        page.on('console', msg => console.log('PAGE LOG:', msg.text()));
        const mAP = await page.evaluate(async () => {
            const human = document.querySelector('#human');
            const seg = document.querySelector('#seg');
            const back_canvas = document.getElementById('back_canvas');
            const back_ctx = back_canvas.getContext('2d');

            const seg_canvas = document.createElement('canvas');
            const seg_ctx = seg_canvas.getContext('2d');
            seg_canvas.width = back_canvas.width = seg.naturalWidth;
            seg_canvas.height = back_canvas.height = seg.naturalHeight;
            seg_ctx.drawImage(seg, 0, 0, seg_canvas.width, seg_canvas.height);

            const humanseg = paddlejs['humanseg'];

            await humanseg.load(true, false, './models/pphumanseg_lite/model.json');
            const {
                data
            } = await humanseg.getGrayValue(human);
            humanseg.drawHumanSeg(data, back_canvas);

            const backImageData = back_ctx.getImageData(0, 0, back_canvas.width, back_canvas.height).data;
            const segImageData = seg_ctx.getImageData(0, 0, seg_canvas.width, seg_canvas.height).data;

            let diffPixelsNum = 0;
            for (let index = 0; index < backImageData.length; index++) {
                if (backImageData[index] !== segImageData[index]) {
                    diffPixelsNum++;
                }
            }
            return diffPixelsNum / backImageData.length;
        });

        const expectedMAP = 0.02;
        await expect(mAP).toBeLessThanOrEqual(expectedMAP);
    });
});