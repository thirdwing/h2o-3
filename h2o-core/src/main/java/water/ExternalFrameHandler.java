package water;

import water.fvec.*;
import water.parser.BufferedString;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.SocketChannel;
import java.util.UUID;

/**
 * Add chunks and data to non-finalized frame from non-h2o environment (ie. Spark executors)
 */
public class ExternalFrameHandler {

    // main tasks
    public static final int CREATE_FRAME = 0;
    public static final int DOWNLOAD_FRAME = 1;

    // subtaks for task CREATE_FRAME
    public static final int CREATE_NEW_CHUNK = 2;
    public static final int ADD_TO_FRAME = 3;
    public static final int CLOSE_NEW_CHUNK = 4;

    public static final int TYPE_NUM = 1;
    public static final int TYPE_STR = 2;
    public static final int TYPE_NA = 3;

    // hints for expected types in order to handle download properly
    public static final byte T_INTEGER = 0;
    public static final byte T_DOUBLE = 1;
    public static final byte T_STRING = 2;



    public void process(AutoBuffer ab, SocketChannel sock) {
        // skip 2 bytes for port set by ab.putUdp. The port is
        // is zero anyway because the request came from non-h2o node and zero is default value
        ab.getPort();

        int requestType = ab.getInt();
        switch (requestType) {
            case CREATE_FRAME:
                handleCreateFrame(ab);
                break;
            case DOWNLOAD_FRAME:
                handleDownloadFrame(ab, sock);
                break;
        }
    }


    private void handleDownloadFrame(AutoBuffer recvAb, SocketChannel sock) {
        String frame_key = recvAb.getStr();
        byte[] expectedTypes = recvAb.getA1();
        assert expectedTypes!=null;
        int chunk_id = recvAb.getInt();

        Frame fr = DKV.getGet(frame_key);

        Chunk[] chunks = ChunkUtils.getChunks(fr, chunk_id);
        AutoBuffer ab = new AutoBuffer();
        ab.putUdp(UDP.udp.external_frame);
        ab.putInt(chunks[0]._len); // num of rows
        writeToChannel(ab, sock);
        ab.flipForReading();

        for (int rowIdx = 0; rowIdx < chunks[0]._len; rowIdx++) { // for each row
            for (int cidx = 0; cidx < chunks.length; cidx++) { // go through the chunks
                ab.clearForWriting(H2O.MAX_PRIORITY); // reuse existing ByteBuffer
                // write flag weather the row is na or not
                if (chunks[cidx].isNA(rowIdx)) {
                    ab.putInt(1);
                } else {
                    ab.putInt(0);

                    Chunk chnk = chunks[cidx];
                    switch (expectedTypes[cidx]){
                        case T_INTEGER:
                            if(chnk.vec().isNumeric() || chnk.vec().isTime()){
                                ab.put8(chnk.at8(rowIdx));
                            }else{
                                assert chnk.vec().domain()!=null && chnk.vec().domain().length!=0;
                                // in this case the chunk is categorical with integers in the
                                // domain
                                ab.put8(Integer.parseInt(chnk.vec().domain()[(int) chnk.at8(rowIdx)]));
                            }
                            break;
                        case T_DOUBLE:
                            assert chnk.vec().isNumeric();
                            if(chnk.vec().isInt()){
                                ab.put8(chnk.at8(rowIdx));
                            }else{
                                ab.put8d(chnk.atd(rowIdx));
                            }

                            break;
                        case T_STRING:
                            assert chnk.vec().isCategorical() || chnk.vec().isString() || chnk.vec().isUUID();
                            ab.putStr(getStringFromChunk(chunks, cidx, rowIdx));
                            break;
                    }

                }
                writeToChannel(ab, sock);
            }
        }
    }

    private void handleCreateFrame(AutoBuffer ab) {
        NewChunk[] nchnk = null;
        int requestType;
        do {
            requestType = ab.getInt();
            switch (requestType) {
                case CREATE_NEW_CHUNK: // Create new chunks
                    String frame_key = ab.getStr();
                    byte[] vec_types = ab.getA1();
                    int chunk_id = ab.getInt();
                    nchnk = ChunkUtils.createNewChunks(frame_key, vec_types, chunk_id);
                    break;
                case ADD_TO_FRAME: // Add to existing frame
                    int dataType = ab.getInt();
                    int colNum = ab.getInt();
                    assert nchnk != null;
                    switch (dataType) {
                        case TYPE_NA:
                            nchnk[colNum].addNA();
                            break;
                        case TYPE_NUM:
                            double d = ab.get8d();
                            nchnk[colNum].addNum(d);
                            break;
                        case TYPE_STR:
                            String str = ab.getStr();
                            // Helper to hold H2O string
                            nchnk[colNum].addStr(new BufferedString(str));
                            break;
                    }
                    break;
                case CLOSE_NEW_CHUNK: // Close new chunks
                    ChunkUtils.closeNewChunks(nchnk);
                    break;
            }
        } while (requestType != CLOSE_NEW_CHUNK);
    }

    private void writeToChannel(AutoBuffer ab, SocketChannel channel) {
        try {
            ab._bb.flip();
            while (ab._bb.hasRemaining()) {
                channel.write(ab._bb);
            }
        } catch (IOException ignore) {
            //TODO: Handle this exception
        }
    }

    private String getStringFromChunk(Chunk[] chks, int columnNum, int rowIdx) {
        if (chks[columnNum].vec().isCategorical()) {
            return chks[columnNum].vec().domain()[(int) chks[columnNum].at8(rowIdx)];
        } else if (chks[columnNum].vec().isString()) {
            BufferedString valStr = new BufferedString();
            chks[columnNum].atStr(valStr, rowIdx); // TODO improve this.
            return valStr.toString();
        } else if (chks[columnNum].vec().isUUID()) {
            UUID uuid = new UUID(chks[columnNum].at16h(rowIdx), chks[columnNum].at16l(rowIdx));
            return uuid.toString();
        } else {
            assert false; // should never return null
            return null; // To make Java Compiler happy
        }
    }


    // Handle the remote-side incoming UDP packet.  This is called on the REMOTE
    // Node, not local.  Wrong thread, wrong JVM.
    static class Adder extends UDP {
        @Override
        AutoBuffer call(AutoBuffer ab) {
            throw H2O.fail();
        }
    }
}
